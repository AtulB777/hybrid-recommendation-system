"""Feature engineering: user, item and interaction features + the sparse user-item matrix.

Everything is computed *as of* a timestamp, from the interactions it is given. The evaluation
code passes training interactions only, so no feature can leak held-out behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_HALF_LIFE_DAYS = 120.0


def parse_genres(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [g.strip() for g in str(value).split("|") if g.strip()]


# ------------------------------------------------------------------------ user context
@dataclass
class UserContext:
    """A user's interaction vector (1 x n_items, time-decayed weights) at scoring time.

    `row_index` is the user's row in the training matrix (None for anonymous/new users).
    User-based CF needs it to avoid treating the user as their own nearest neighbour.
    """

    vector: sp.csr_matrix
    row_index: int | None = None

    @property
    def seen_idx(self) -> np.ndarray:
        return self.vector.indices

    @property
    def n_history(self) -> int:
        return int(self.vector.nnz)

    @property
    def is_cold(self) -> bool:
        return self.vector.nnz == 0

    @staticmethod
    def empty(n_items: int) -> "UserContext":
        return UserContext(sp.csr_matrix((1, n_items), dtype=np.float32))


# ------------------------------------------------------------- interaction features
def add_interaction_features(
    interactions: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float
) -> pd.DataFrame:
    """Adds age_days, decay = 0.5**(age/half_life) and weight_decayed = weight * decay."""
    out = interactions.copy()
    age = (as_of - pd.to_datetime(out["timestamp"])).dt.total_seconds() / 86400.0
    out["age_days"] = age.clip(lower=0.0)
    out["decay"] = np.power(0.5, out["age_days"] / half_life_days)
    out["weight_decayed"] = out["weight"] * out["decay"]
    return out


# --------------------------------------------------------------------- item features
def build_item_features(items: pd.DataFrame, inter_f: pd.DataFrame) -> pd.DataFrame:
    items = items.sort_values("item_id").set_index("item_id")
    if len(inter_f):
        g = inter_f.groupby("item_id")
        agg = g.agg(
            n_interactions=("user_id", "size"),
            popularity_score=("weight_decayed", "sum"),
            avg_weight=("weight", "mean"),
            last_age_days=("age_days", "min"),
        )
        recent = inter_f[inter_f["age_days"] <= 30].groupby("item_id").size().rename("n_recent_30d")
        purchase = (
            inter_f.assign(p=(inter_f["event_type"] == "purchase").astype(float))
            .groupby("item_id")["p"].mean().rename("purchase_rate")
        )
        feats = items.join([agg, recent, purchase])
    else:
        feats = items.copy()
        for c in ("n_interactions", "popularity_score", "avg_weight", "last_age_days", "n_recent_30d", "purchase_rate"):
            feats[c] = 0.0
    for c in ("n_interactions", "popularity_score", "avg_weight", "n_recent_30d", "purchase_rate"):
        feats[c] = feats[c].fillna(0.0)
    feats["last_age_days"] = feats["last_age_days"].fillna(np.inf)
    feats["popularity_pct"] = feats["popularity_score"].rank(pct=True)
    feats["is_cold"] = feats["n_interactions"] == 0
    feats["genres_list"] = feats["genres"].map(parse_genres)
    genre_text = feats["genres"].str.replace("|", " ", regex=False).str.replace("-", "", regex=False)
    # genre tokens are repeated so they carry more TF-IDF weight than a single description word
    feats["text"] = feats["title"].astype(str) + " " + genre_text + " " + genre_text + " " + feats["description"].fillna("")
    return feats


# --------------------------------------------------------------------- user features
def build_user_features(users: pd.DataFrame, inter_f: pd.DataFrame, items_f: pd.DataFrame) -> pd.DataFrame:
    users = users.sort_values("user_id").set_index("user_id")
    if not len(inter_f):
        users["n_interactions"] = 0
        return users
    g = inter_f.groupby("user_id")
    agg = g.agg(
        n_interactions=("item_id", "size"),
        avg_weight=("weight", "mean"),
        days_since_last=("age_days", "min"),
        activity_span_days=("age_days", "max"),
    )
    purchases = (
        inter_f.assign(p=(inter_f["event_type"] == "purchase").astype(float)).groupby("user_id")["p"].mean()
    ).rename("purchase_rate")

    exploded = (
        inter_f[["user_id", "item_id", "weight_decayed"]]
        .merge(items_f[["genres_list"]], left_on="item_id", right_index=True)
        .explode("genres_list")
    )
    aff = exploded.groupby(["user_id", "genres_list"])["weight_decayed"].sum().unstack(fill_value=0.0)
    aff = aff.div(aff.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    top_genre = aff.idxmax(axis=1).rename("top_genre")
    n_genres = (aff > 0).sum(axis=1).rename("n_distinct_genres")
    aff = aff.add_prefix("aff_")

    feats = users.join([agg, purchases, top_genre, n_genres, aff])
    feats["n_interactions"] = feats["n_interactions"].fillna(0).astype(int)
    feats["is_cold"] = feats["n_interactions"] == 0
    return feats


# ------------------------------------------------------------------- feature bundle
@dataclass
class Features:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict[int, int]
    item_index: dict[int, int]
    matrix: sp.csr_matrix  # users x items, time-decayed implicit feedback strength
    items: pd.DataFrame  # indexed by item_id, aligned with item_ids
    users: pd.DataFrame  # indexed by user_id, aligned with user_ids
    as_of: pd.Timestamp
    half_life_days: float

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def context_for_user(self, user_id: int) -> UserContext:
        idx = self.user_index.get(int(user_id))
        if idx is None:
            return UserContext.empty(self.n_items)
        return UserContext(self.matrix[idx].copy(), idx)

    def context_from_history(
        self,
        history: pd.DataFrame | None,
        user_id: int | None = None,
        extra_item_ids: Iterable[int] = (),
    ) -> UserContext:
        """Build a context from *live* interactions (DB rows) and/or ad-hoc item ids.

        Used online so a user's newest activity is reflected without retraining.
        """
        weights: dict[int, float] = {}
        if history is not None and len(history):
            h = add_interaction_features(history, self.as_of, self.half_life_days)
            for item_id, w in zip(h["item_id"], h["weight_decayed"]):
                j = self.item_index.get(int(item_id))
                if j is not None:
                    weights[j] = max(weights.get(j, 0.0), float(w))
        for item_id in extra_item_ids:
            j = self.item_index.get(int(item_id))
            if j is not None:
                weights[j] = max(weights.get(j, 0.0), 1.0)
        cols = np.fromiter(weights.keys(), dtype=np.int64, count=len(weights))
        data = np.fromiter(weights.values(), dtype=np.float32, count=len(weights))
        vec = sp.csr_matrix((data, (np.zeros(len(cols), dtype=np.int64), cols)), shape=(1, self.n_items))
        row = self.user_index.get(int(user_id)) if user_id is not None else None
        return UserContext(vec, row)


def build_features(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> Features:
    inter = interactions.copy()
    inter["timestamp"] = pd.to_datetime(inter["timestamp"])
    if as_of is None:
        as_of = inter["timestamp"].max() if len(inter) else pd.Timestamp.now()
    as_of = pd.Timestamp(as_of)

    inter_f = add_interaction_features(inter, as_of, half_life_days)
    items_f = build_item_features(items, inter_f)
    users_f = build_user_features(users, inter_f, items_f)

    item_ids = items_f.index.to_numpy()
    user_ids = users_f.index.to_numpy()
    item_index = {int(i): n for n, i in enumerate(item_ids)}
    user_index = {int(u): n for n, u in enumerate(user_ids)}

    rows = inter_f["user_id"].map(user_index)
    cols = inter_f["item_id"].map(item_index)
    ok = rows.notna() & cols.notna()
    matrix = sp.csr_matrix(
        (
            inter_f.loc[ok, "weight_decayed"].to_numpy(dtype=np.float32),
            (rows[ok].to_numpy(dtype=np.int64), cols[ok].to_numpy(dtype=np.int64)),
        ),
        shape=(len(user_ids), len(item_ids)),
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return Features(user_ids, item_ids, user_index, item_index, matrix, items_f, users_f, as_of, half_life_days)

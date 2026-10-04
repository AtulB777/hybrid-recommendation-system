from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from app.features.engineering import Features, UserContext
from app.recommenders.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering on the user-item matrix.

    item_sim = cosine similarity of item columns, shrunk by co-occurrence support
    (sim * c / (c + shrinkage)) so pairs seen together once don't look as reliable as pairs
    seen together fifty times, then pruned to the `k_neighbors` strongest neighbours per item.
    score(u, i) = sum_j  r_uj * sim(j, i).
    """

    name = "cf_item"
    reason_type = "collaborative"
    description = "Item-based collaborative filtering: cosine item-item similarity from the user-item matrix, with support shrinkage."

    def __init__(self, k_neighbors: int = 50, shrinkage: float = 5.0) -> None:
        super().__init__()
        self.k_neighbors, self.shrinkage = k_neighbors, shrinkage

    def params(self) -> dict:
        return {"k_neighbors": self.k_neighbors, "shrinkage": self.shrinkage}

    def fit(self, feats: Features) -> "ItemKNNRecommender":
        super().fit(feats)
        R = feats.matrix
        B = (R > 0).astype(np.float32)
        cooc = (B.T @ B).toarray()
        Rn = normalize(R.T.tocsr(), norm="l2", axis=1)
        S = (Rn @ Rn.T).toarray().astype(np.float32)
        np.fill_diagonal(S, 0.0)
        S *= cooc / (cooc + self.shrinkage)
        n = S.shape[0]
        if self.k_neighbors < n - 1:
            drop = np.argpartition(-S, self.k_neighbors, axis=1)[:, self.k_neighbors:]
            np.put_along_axis(S, drop, 0.0, axis=1)
        self.S = S  # dense: fine to ~20k items; see README for the sparse top-N variant
        return self

    def score(self, ctx: UserContext) -> np.ndarray:
        if ctx.is_cold:
            return np.zeros(self.feats.n_items)
        return np.asarray(ctx.vector @ self.S).ravel()

    def supporting_items(self, ctx: UserContext, item_idx: int, n: int = 2) -> list[int]:
        seen = ctx.seen_idx
        if not len(seen):
            return []
        contrib = self.S[seen, item_idx] * ctx.vector.data
        order = np.argsort(-contrib, kind="stable")[:n]
        return [int(seen[i]) for i in order if contrib[i] > 0]

    def item_similarity_row(self, item_idx: int) -> np.ndarray:
        return self.S[item_idx]

    def similar_items(self, item_idx: int, k: int = 10) -> list[tuple[int, float]]:
        row = self.S[item_idx].astype(np.float64)
        order = np.argsort(-row, kind="stable")[:k]
        return [(int(j), float(row[j])) for j in order if row[j] > 0]


class UserKNNRecommender(BaseRecommender):
    """User-based collaborative filtering: score items by what the most similar users liked.

    sim(u, v) = cosine of interaction vectors; keep the `k_neighbors` most similar users and
    score(u, i) = sum_v sim(u, v) * r_vi.
    """

    name = "cf_user"
    reason_type = "collaborative"
    description = "User-based collaborative filtering: weighted vote of the k most similar users' interactions."

    def __init__(self, k_neighbors: int = 40) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors

    def params(self) -> dict:
        return {"k_neighbors": self.k_neighbors}

    def fit(self, feats: Features) -> "UserKNNRecommender":
        super().fit(feats)
        self.R = feats.matrix.tocsr()
        self.Rn = normalize(self.R, norm="l2", axis=1)
        return self

    def _neighbors(self, ctx: UserContext) -> tuple[np.ndarray, np.ndarray]:
        un = normalize(ctx.vector, norm="l2")
        sims = np.asarray((self.Rn @ un.T).todense()).ravel()
        if ctx.row_index is not None:
            sims[ctx.row_index] = 0.0  # never use yourself as a neighbour
        k = min(self.k_neighbors, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[sims[top] > 0]
        return top, sims[top]

    def score(self, ctx: UserContext) -> np.ndarray:
        if ctx.is_cold:
            return np.zeros(self.feats.n_items)
        nb, w = self._neighbors(ctx)
        if not len(nb):
            return np.zeros(self.feats.n_items)
        return np.asarray(self.R[nb].T @ w).ravel()

    def supporting_items(self, ctx: UserContext, item_idx: int, n: int = 2) -> list[int]:
        seen = ctx.seen_idx
        if not len(seen):
            return []
        nb, w = self._neighbors(ctx)
        if not len(nb):
            return []
        has_item = np.asarray(self.R[nb][:, item_idx].todense()).ravel() > 0
        nb, w = nb[has_item], w[has_item]
        if not len(nb):
            return []
        # seen items that the neighbours who liked `item_idx` also interacted with
        overlap = np.asarray((self.R[nb][:, seen].T @ w)).ravel() * ctx.vector.data
        order = np.argsort(-overlap, kind="stable")[:n]
        return [int(seen[i]) for i in order if overlap[i] > 0]

"""Data ingestion: CSV -> validation/cleaning -> relational database (and back).

Expected CSV schemas
--------------------
users.csv         user_id [int], country, age_group, signup_date (all but user_id optional)
items.csv         item_id [int], title, genres ("a|b"), description, release_year (all but item_id optional)
interactions.csv  user_id, item_id, timestamp, event_type (view|like|purchase), weight (optional)

Cleaning rules (every drop is counted in the returned report, nothing is silent):
  * missing / unparsable keys or timestamps are dropped
  * duplicate users/items are collapsed (first wins)
  * interactions referencing unknown users/items are dropped (referential integrity)
  * duplicate (user, item) pairs are collapsed to ONE row holding the strongest event and the
    latest timestamp - the models consume implicit feedback strength, not event logs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.engine import Engine

from app.models.orm import Interaction, Item, Recommendation, User
from app.services.synthetic_data import EVENT_WEIGHTS

REQUIRED = {
    "users": ["user_id"],
    "items": ["item_id", "title", "genres"],
    "interactions": ["user_id", "item_id", "timestamp"],
}


@dataclass
class DataBundle:
    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame


@dataclass
class IngestionReport:
    dropped: dict[str, int] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)

    def _drop(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def summary(self) -> str:
        kept = ", ".join(f"{k}={v}" for k, v in self.kept.items())
        dropped = ", ".join(f"{k}={v}" for k, v in self.dropped.items()) or "nothing"
        return f"kept[{kept}] dropped[{dropped}]"


def _require(df: pd.DataFrame, table: str) -> None:
    missing = [c for c in REQUIRED[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns {missing}")


def validate_bundle(bundle: DataBundle) -> tuple[DataBundle, IngestionReport]:
    report = IngestionReport()
    for name in REQUIRED:
        _require(getattr(bundle, name), name)

    # ---- users ----
    users = bundle.users.copy()
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce")
    n0 = len(users)
    users = users.dropna(subset=["user_id"])
    report._drop("users_bad_key", n0 - len(users))
    users["user_id"] = users["user_id"].astype("int64")
    n0 = len(users)
    users = users.drop_duplicates("user_id", keep="first")
    report._drop("users_duplicate", n0 - len(users))
    for col in ("country", "age_group", "signup_date"):
        if col not in users.columns:
            users[col] = None
    users["signup_date"] = pd.to_datetime(users["signup_date"], errors="coerce")

    # ---- items ----
    items = bundle.items.copy()
    items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce")
    n0 = len(items)
    items = items.dropna(subset=["item_id", "title"])
    report._drop("items_bad_key_or_title", n0 - len(items))
    items["item_id"] = items["item_id"].astype("int64")
    n0 = len(items)
    items = items.drop_duplicates("item_id", keep="first")
    report._drop("items_duplicate", n0 - len(items))
    items["genres"] = items["genres"].fillna("unknown").astype(str)
    if "description" not in items.columns:
        items["description"] = ""
    items["description"] = items["description"].fillna("").astype(str)
    if "release_year" not in items.columns:
        items["release_year"] = None
    items["release_year"] = pd.to_numeric(items["release_year"], errors="coerce").astype("Int64")

    # ---- interactions ----
    inter = bundle.interactions.copy()
    inter["user_id"] = pd.to_numeric(inter["user_id"], errors="coerce")
    inter["item_id"] = pd.to_numeric(inter["item_id"], errors="coerce")
    inter["timestamp"] = pd.to_datetime(inter["timestamp"], errors="coerce")
    n0 = len(inter)
    inter = inter.dropna(subset=["user_id", "item_id", "timestamp"])
    report._drop("interactions_bad_row", n0 - len(inter))
    inter[["user_id", "item_id"]] = inter[["user_id", "item_id"]].astype("int64")

    if "event_type" not in inter.columns:
        inter["event_type"] = "view"
    inter["event_type"] = inter["event_type"].fillna("view").astype(str)
    if "weight" not in inter.columns:
        inter["weight"] = inter["event_type"].map(EVENT_WEIGHTS).fillna(1.0)
    inter["weight"] = pd.to_numeric(inter["weight"], errors="coerce")
    n0 = len(inter)
    inter = inter[inter["weight"] > 0]
    report._drop("interactions_nonpositive_weight", n0 - len(inter))

    n0 = len(inter)
    inter = inter[inter["user_id"].isin(users["user_id"]) & inter["item_id"].isin(items["item_id"])]
    report._drop("interactions_orphan", n0 - len(inter))

    n0 = len(inter)
    latest = inter.groupby(["user_id", "item_id"])["timestamp"].transform("max")
    inter = inter.assign(timestamp=latest).sort_values(["weight"], kind="stable")
    inter = inter.drop_duplicates(["user_id", "item_id"], keep="last")
    report._drop("interactions_duplicate_pair", n0 - len(inter))

    inter = inter[["user_id", "item_id", "event_type", "weight", "timestamp"]]
    inter = inter.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    users = users.sort_values("user_id").reset_index(drop=True)
    items = items.sort_values("item_id").reset_index(drop=True)

    report.kept = {"users": len(users), "items": len(items), "interactions": len(inter)}
    return DataBundle(users, items, inter), report


# ---------------------------------------------------------------------------- CSV I/O
def write_csv_bundle(bundle: DataBundle, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle.users.to_csv(directory / "users.csv", index=False)
    bundle.items.to_csv(directory / "items.csv", index=False)
    bundle.interactions.to_csv(directory / "interactions.csv", index=False)


def load_csv_bundle(directory: Path) -> DataBundle:
    directory = Path(directory)
    paths = {n: directory / f"{n}.csv" for n in REQUIRED}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Generate data with `python -m data.generate_synthetic`."
        )
    return DataBundle(
        users=pd.read_csv(paths["users"]),
        items=pd.read_csv(paths["items"]),
        interactions=pd.read_csv(paths["interactions"]),
    )


# ------------------------------------------------------------------------ database I/O
def ingest_to_db(bundle: DataBundle, engine: Engine) -> None:
    """Replace users/items/interactions (and stale recommendations) with `bundle`."""
    with engine.begin() as conn:
        # children first to respect foreign keys on PostgreSQL
        for model in (Recommendation, Interaction, Item, User):
            conn.execute(delete(model))
        bundle.users.to_sql("users", conn, if_exists="append", index=False, chunksize=500)
        items = bundle.items.copy()
        items["release_year"] = items["release_year"].astype(object).where(items["release_year"].notna(), None)
        items.to_sql("items", conn, if_exists="append", index=False, chunksize=500)
        bundle.interactions.to_sql("interactions", conn, if_exists="append", index=False, chunksize=500)


def load_bundle_from_db(engine: Engine) -> DataBundle:
    with engine.connect() as conn:
        users = pd.read_sql(text("SELECT user_id, country, age_group, signup_date FROM users"), conn)
        items = pd.read_sql(text("SELECT item_id, title, genres, description, release_year FROM items"), conn)
        inter = pd.read_sql(
            text("SELECT user_id, item_id, event_type, weight, timestamp FROM interactions"), conn
        )
    inter["timestamp"] = pd.to_datetime(inter["timestamp"])
    bundle, _ = validate_bundle(DataBundle(users, items, inter))
    return bundle

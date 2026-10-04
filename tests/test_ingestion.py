import pandas as pd
import pytest

from app.database.session import init_db, make_engine
from app.services.ingestion import (
    DataBundle, ingest_to_db, load_bundle_from_db, validate_bundle,
)


def _dirty():
    users = pd.DataFrame({"user_id": [1, 2, 2, None]})
    items = pd.DataFrame({"item_id": [10, 11, 11], "title": ["a", "b", "b2"], "genres": ["x", "y|z", "y"]})
    inter = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 9, 2, 1],
            "item_id": [10, 10, 11, 11, 10, 99, 11],
            "timestamp": ["2026-01-01", "2026-02-01", "not-a-date", "2026-01-05", "2026-01-05", "2026-01-06", "2026-03-01"],
            "event_type": ["view", "purchase", "view", "like", "view", "view", "view"],
            "weight": [1, 4, 1, 2, 1, 1, -1],
        }
    )
    return DataBundle(users, items, inter)


def test_validation_cleans_and_reports_every_drop():
    clean, rep = validate_bundle(_dirty())
    assert rep.dropped["users_bad_key"] == 1
    assert rep.dropped["users_duplicate"] == 1
    assert rep.dropped["items_duplicate"] == 1
    assert rep.dropped["interactions_bad_row"] == 1  # bad timestamp
    assert rep.dropped["interactions_nonpositive_weight"] == 1
    assert rep.dropped["interactions_orphan"] == 2  # user 9, item 99
    assert rep.dropped["interactions_duplicate_pair"] == 1
    assert set(clean.users["user_id"]) == {1, 2}
    # (1,10) seen twice: strongest event (purchase) kept with the latest timestamp
    row = clean.interactions.query("user_id == 1 and item_id == 10").iloc[0]
    assert row["event_type"] == "purchase" and row["weight"] == 4
    assert row["timestamp"] == pd.Timestamp("2026-02-01")


def test_missing_required_columns_raises():
    bad = DataBundle(pd.DataFrame({"id": [1]}), _dirty().items, _dirty().interactions)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_bundle(bad)


def test_default_weight_from_event_type():
    d = _dirty()
    d.interactions = d.interactions.drop(columns="weight")
    clean, _ = validate_bundle(d)
    assert clean.interactions.query("event_type == 'purchase'")["weight"].iloc[0] == 4.0


def test_db_roundtrip_and_replace(data):
    eng = make_engine("sqlite://")
    init_db(eng)
    ingest_to_db(data, eng)
    ingest_to_db(data, eng)  # idempotent: replaces, never duplicates
    back = load_bundle_from_db(eng)
    assert len(back.users) == len(data.users)
    assert len(back.items) == len(data.items)
    assert len(back.interactions) == len(data.interactions)

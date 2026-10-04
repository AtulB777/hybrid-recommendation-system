import numpy as np
import pandas as pd
import pytest

from app.evaluation.runner import temporal_split
from app.features.engineering import add_interaction_features, build_features


def test_matrix_shape_and_alignment(data):
    f = build_features(data.users, data.items, data.interactions)
    assert f.matrix.shape == (len(data.users), len(data.items))
    assert f.matrix.nnz == len(data.interactions)
    assert list(f.users.index) == list(f.user_ids) and list(f.items.index) == list(f.item_ids)


def test_recency_decay_halves_at_half_life():
    t = pd.Timestamp("2026-06-30")
    df = pd.DataFrame({"timestamp": [t, t - pd.Timedelta(days=100)], "weight": [2.0, 2.0]})
    out = add_interaction_features(df, t, half_life_days=100)
    assert out["weight_decayed"].iloc[0] == pytest.approx(2.0)
    assert out["weight_decayed"].iloc[1] == pytest.approx(1.0)


def test_feature_tables_have_expected_columns(data):
    f = build_features(data.users, data.items, data.interactions)
    assert {"n_interactions", "popularity_score", "popularity_pct", "is_cold", "text"} <= set(f.items.columns)
    assert {"n_interactions", "top_genre", "n_distinct_genres", "is_cold"} <= set(f.users.columns)
    assert any(c.startswith("aff_") for c in f.users.columns)
    aff = f.users.filter(regex="^aff_").sum(axis=1)
    warm = f.users["n_interactions"] > 0
    assert np.allclose(aff[warm], 1.0)  # genre affinities are a distribution
    assert f.items["is_cold"].sum() == 8 and f.users["is_cold"].sum() == 5  # planted cold entities


def test_features_built_from_train_never_see_test(data):
    train, _, test = temporal_split(data.interactions)
    f = build_features(data.users, data.items, train)
    for row in test.head(200).itertuples():
        assert f.matrix[f.user_index[row.user_id], f.item_index[row.item_id]] == 0
    assert f.as_of <= train["timestamp"].max()


def test_context_from_history_matches_matrix(data):
    f = build_features(data.users, data.items, data.interactions)
    uid = int(f.users[f.users["n_interactions"] > 3].index[0])
    hist = data.interactions[data.interactions["user_id"] == uid]
    live = f.context_from_history(hist, uid)
    stored = f.context_for_user(uid)
    assert np.allclose(live.vector.toarray(), stored.vector.toarray(), atol=1e-4)
    assert live.row_index == stored.row_index


def test_context_from_extra_items_and_unknown_ids(model_bundle):
    f = model_bundle.feats
    ctx = f.context_from_history(None, None, extra_item_ids=[int(f.item_ids[0]), 10_000_000])
    assert ctx.n_history == 1 and ctx.row_index is None


def test_unknown_user_gets_cold_context(model_bundle):
    assert model_bundle.feats.context_for_user(123456789).is_cold

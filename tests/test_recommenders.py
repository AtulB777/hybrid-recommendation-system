import re

import numpy as np
import pytest

from app.recommenders.base import top_k_indices

MODELS = ["popularity", "content", "cf_item", "cf_user", "hybrid"]
SUPPORT = re.compile(r'^Recommended because you interacted with "[^"]+"( and "[^"]+")?\.$')


def test_top_k_indices_ties_and_exclusion():
    s = np.array([0.5, 0.9, 0.9, 0.1, -np.inf])
    assert top_k_indices(s, 3).tolist() == [1, 2, 0]  # tie -> lower index first
    assert top_k_indices(s, 3, exclude=np.array([1])).tolist() == [2, 0, 3]
    assert top_k_indices(s, 10).tolist() == [1, 2, 0, 3]  # -inf never returned
    assert top_k_indices(np.zeros(3), 0).size == 0


@pytest.mark.parametrize("name", MODELS)
def test_recommendations_are_valid(model_bundle, warm_user, name):
    f, m = model_bundle.feats, model_bundle.models[name]
    ctx = f.context_for_user(warm_user)
    recs = m.recommend(ctx, 10)
    assert len(recs) == 10
    ids = [r.item_id for r in recs]
    assert len(set(ids)) == 10
    seen = {int(f.item_ids[j]) for j in ctx.seen_idx}
    assert not seen & set(ids), "already-seen items must be excluded"
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in recs] == list(range(1, 11))
    assert all(r.explanation for r in recs)


@pytest.mark.parametrize("name", MODELS)
def test_exclude_seen_false_can_return_seen(model_bundle, warm_user, name):
    f, m = model_bundle.feats, model_bundle.models[name]
    ctx = f.context_for_user(warm_user)
    recs = m.recommend(ctx, f.n_items, exclude_seen=False, explain=False)
    assert len(recs) >= len(m.recommend(ctx, f.n_items, exclude_seen=True, explain=False))


def test_popularity_is_not_personalised(model_bundle):
    f, m = model_bundle.feats, model_bundle.models["popularity"]
    a = [r.item_id for r in m.recommend(f.context_for_user(int(f.user_ids[0])), 10, exclude_seen=False)]
    b = [r.item_id for r in m.recommend(f.context_for_user(int(f.user_ids[1])), 10, exclude_seen=False)]
    assert a == b


def test_personalised_models_differ_across_users(model_bundle):
    f = model_bundle.feats
    warm = f.users[f.users["n_interactions"] >= 10].index[:2]
    for name in ("content", "cf_item", "cf_user"):
        m = model_bundle.models[name]
        a = [r.item_id for r in m.recommend(f.context_for_user(int(warm[0])), 10, explain=False)]
        b = [r.item_id for r in m.recommend(f.context_for_user(int(warm[1])), 10, explain=False)]
        assert a != b


def test_cold_user_gets_popularity_from_hybrid(model_bundle, cold_user):
    f = model_bundle.feats
    ctx = f.context_for_user(cold_user)
    assert ctx.is_cold
    hybrid = model_bundle.models["hybrid"].recommend(ctx, 10)
    pop = model_bundle.models["popularity"].recommend(ctx, 10)
    assert [r.item_id for r in hybrid] == [r.item_id for r in pop]
    assert all(r.reason_type == "popularity" for r in hybrid)
    assert all("popular" in r.explanation for r in hybrid)


def test_effective_weights_policy(model_bundle):
    h = model_bundle.models["hybrid"]
    assert h.effective_weights(0) == {"popularity": 1.0, "content": 0.0, "cf_item": 0.0, "cf_user": 0.0}
    for n in (1, 2, 3, 50):
        w = h.effective_weights(n)
        assert sum(w.values()) == pytest.approx(1.0)
    # short histories lean less on collaborative signals than long ones
    assert h.effective_weights(1)["cf_item"] < h.effective_weights(50)["cf_item"]
    assert h.effective_weights(50) == pytest.approx(h.weights)


def test_hybrid_weight_validation(model_bundle):
    h = model_bundle.models["hybrid"]
    with pytest.raises(ValueError):
        h._normalise_weights({"popularity": -1}, h.components)
    with pytest.raises(ValueError):
        h._normalise_weights({}, h.components)


def test_explanations_name_real_history_items(model_bundle, warm_user):
    f = model_bundle.feats
    ctx = f.context_for_user(warm_user)
    history_titles = {f.items["title"].iloc[j] for j in ctx.seen_idx}
    recs = model_bundle.models["hybrid"].recommend(ctx, 10)
    personalised = [r for r in recs if r.reason_type != "popularity"]
    assert personalised, "a user with history should get personalised reasons"
    for r in personalised:
        assert SUPPORT.match(r.explanation), r.explanation
        cited = re.findall(r'"([^"]+)"', r.explanation)
        assert set(cited) <= history_titles  # never cites an item the user didn't interact with


def test_content_scores_cold_items_but_cf_cannot(model_bundle, warm_user):
    f = model_bundle.feats
    cold_idx = np.flatnonzero(f.items["is_cold"].to_numpy())
    ctx = f.context_for_user(warm_user)
    assert (model_bundle.models["content"].score(ctx)[cold_idx] > 0).any()
    assert (model_bundle.models["cf_item"].score(ctx)[cold_idx] == 0).all()
    assert (model_bundle.models["cf_user"].score(ctx)[cold_idx] == 0).all()


def test_hybrid_can_surface_cold_items(model_bundle):
    f = model_bundle.feats
    cold_ids = set(map(int, f.item_ids[f.items["is_cold"].to_numpy()]))
    hybrid = model_bundle.models["hybrid"]
    surfaced = set()
    for uid in f.users[f.users["n_interactions"] > 0].index[:100]:
        surfaced |= {r.item_id for r in hybrid.recommend(f.context_for_user(int(uid)), 20, explain=False)}
    assert surfaced & cold_ids, "content signal should let new items reach some users"


def test_user_knn_never_uses_self_as_neighbour(model_bundle, warm_user):
    m = model_bundle.models["cf_user"]
    ctx = model_bundle.feats.context_for_user(warm_user)
    nb, _ = m._neighbors(ctx)
    assert ctx.row_index not in nb


def test_similar_items(model_bundle):
    f, h = model_bundle.feats, model_bundle.models["hybrid"]
    warm_idx = int(np.flatnonzero(~f.items["is_cold"].to_numpy())[0])
    cold_idx = int(np.flatnonzero(f.items["is_cold"].to_numpy())[0])
    for method in ("hybrid", "content", "collaborative"):
        res = h.similar_items(warm_idx, 5, method)
        assert warm_idx not in [j for j, _ in res]
        sims = [s for _, s in res]
        assert sims == sorted(sims, reverse=True)
    assert h.similar_items(cold_idx, 5, "hybrid"), "cold item falls back to content similarity"
    assert h.similar_items(cold_idx, 5, "collaborative") == []
    with pytest.raises(ValueError):
        h.similar_items(warm_idx, 5, "bogus")

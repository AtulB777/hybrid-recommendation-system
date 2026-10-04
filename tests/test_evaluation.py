import numpy as np
import pandas as pd
import pytest

from app.evaluation.runner import (
    evaluate_model, relevant_map, run_comparison, temporal_split, tune_hybrid_weights,
)
from app.features.engineering import build_features
from app.recommenders.base import BaseRecommender


def test_split_is_disjoint_complete_and_temporal(data):
    train, val, test = temporal_split(data.interactions)
    assert len(train) + len(val) + len(test) == len(data.interactions)
    keys = lambda d: set(zip(d.user_id, d.item_id))
    assert not keys(train) & keys(val) and not keys(train) & keys(test) and not keys(val) & keys(test)
    g = pd.concat([train.assign(s=0), val.assign(s=1), test.assign(s=2)]).groupby(["user_id", "s"])["timestamp"].agg(["min", "max"]).unstack()
    assert (g[("max", 0)] <= g[("min", 1)]).dropna().all()
    assert (g[("max", 1)] <= g[("min", 2)]).dropna().all()


def test_users_below_min_interactions_stay_in_train(data):
    train, val, test = temporal_split(data.interactions, min_interactions=5)
    counts = data.interactions.groupby("user_id").size()
    small = set(counts[counts < 5].index)
    assert not small & set(test.user_id) and not small & set(val.user_id)


class Oracle(BaseRecommender):
    """Cheats by scoring held-out items highest - proves the evaluator measures what it should."""
    name = "oracle"

    def __init__(self, relevant, item_index):
        super().__init__()
        self.relevant, self.item_index = relevant, item_index

    def score(self, ctx):
        raise NotImplementedError

    def _score_with_state(self, ctx, exclude_seen):
        raise NotImplementedError


def test_evaluator_gives_perfect_score_to_an_oracle(data):
    train, _, test = temporal_split(data.interactions)
    feats = build_features(data.users, data.items, train)
    rel = relevant_map(test)

    class Cheat(Oracle):
        def recommend(self, ctx, k=10, exclude_seen=True, explain=True, exclude_idx=None):
            from app.recommenders.base import ScoredItem
            uid = int(feats.user_ids[ctx.row_index])
            items = sorted(self.relevant[uid])[:k]
            return [ScoredItem(i, self.item_index[i], 1.0, r + 1) for r, i in enumerate(items)]

    res = evaluate_model(Cheat(rel, feats.item_index), feats, rel, [1])
    assert res["n_users"] > 50
    assert res["metrics"]["precision@1"] == 1.0
    assert res["metrics"]["ndcg@1"] == 1.0
    assert res["metrics"]["map@1"] == 1.0


def test_comparison_report_structure_and_sanity(data):
    report = run_comparison(data, ks=(5, 10))
    assert set(report["results"]) == {"random", "popularity", "content", "cf_item", "cf_user", "hybrid"}
    n_users = {r["n_users"] for r in report["results"].values()}
    assert len(n_users) == 1, "all models must be judged on the same users"
    for res in report["results"].values():
        for key, v in res["metrics"].items():
            assert 0.0 <= v <= 1.0, key
        assert res["metrics"]["recall@10"] >= res["metrics"]["recall@5"]
    r = report["results"]
    # real models must beat the random floor on this dataset (fixed seed => deterministic)
    for name in ("popularity", "cf_item", "hybrid"):
        assert r[name]["metrics"]["ndcg@10"] > r["random"]["metrics"]["ndcg@10"], name
    p = report["protocol"]
    assert p["n_test_interactions"] > 0 and p["split"].startswith("per-user temporal")


def test_model_subset_and_unknown_model(data):
    report = run_comparison(data, ks=(5,), model_names=["popularity"], include_random=False)
    assert list(report["results"]) == ["popularity"]
    with pytest.raises(ValueError, match="unknown models"):
        run_comparison(data, ks=(5,), model_names=["nope"])


def test_weight_tuning_returns_valid_distribution(data):
    weights, trials = tune_hybrid_weights(data, k=10, step=0.5, max_popularity=None)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert len(trials) == 10  # compositions of 2 halves into 4 buckets: C(5,3)
    best = max(t["ndcg@10"] for t in trials)
    assert [t for t in trials if t["weights"] == weights][0]["ndcg@10"] == best


def test_popularity_cap_constrains_the_search(data):
    weights, trials = tune_hybrid_weights(data, k=10, step=0.5, max_popularity=0.3)
    assert weights["popularity"] <= 0.3
    assert all(t["weights"]["popularity"] <= 0.3 for t in trials)
    assert len(trials) == 6  # popularity forced to 0 -> compositions of 2 halves into 3 buckets


def test_report_includes_standard_errors(data):
    res = run_comparison(data, ks=(10,), model_names=["hybrid"], include_random=False)["results"]["hybrid"]
    assert res["stderr"]["ndcg@10"] > 0 and set(res["stderr"]) <= set(res["metrics"])

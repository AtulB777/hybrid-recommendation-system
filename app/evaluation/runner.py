"""Offline evaluation protocol shared by `training/evaluate.py`, `training/train.py` and POST /evaluate.

Protocol (temporal, per-user leave-last-out):
  * each user with >= `min_interactions` interactions has their most recent `test_frac` held out
    as TEST and the `val_frac` before that as VALIDATION; everything earlier is TRAIN
  * hybrid weights are tuned on VALIDATION using models fitted on TRAIN
  * the final comparison fits on TRAIN+VALIDATION and is scored on TEST (never touched by tuning)
  * relevant items = held-out items; items in the fit data are excluded from recommendations
  * only users that have fit-time history AND at least one held-out item are scored, so every
    model is judged on the same users (cold users are a serving concern, covered in the README)
"""
from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd

from app.evaluation.metrics import METRICS
from app.features.engineering import DEFAULT_HALF_LIFE_DAYS, Features, build_features
from app.recommenders.base import BaseRecommender
from app.recommenders.random_baseline import RandomRecommender
from app.recommenders.registry import COMPONENT_NAMES, build_models, fit_models
from app.services.ingestion import DataBundle


def temporal_split(
    interactions: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2, min_interactions: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    pos = df.groupby("user_id").cumcount()
    n = df.groupby("user_id")["item_id"].transform("size")
    n_test = np.maximum(1, np.round(n * test_frac)).astype(int)
    n_val = np.maximum(1, np.round(n * val_frac)).astype(int)
    eligible = n >= min_interactions
    from_end = n - 1 - pos  # 0 = most recent
    is_test = eligible & (from_end < n_test)
    is_val = eligible & ~is_test & (from_end < n_test + n_val)
    return df[~is_test & ~is_val], df[is_val], df[is_test]


def relevant_map(df: pd.DataFrame) -> dict[int, set[int]]:
    return {int(u): set(map(int, g["item_id"])) for u, g in df.groupby("user_id")}


def evaluate_model(
    model: BaseRecommender, feats: Features, relevant: dict[int, set[int]], ks: Iterable[int]
) -> dict:
    ks = sorted(set(int(k) for k in ks))
    kmax = max(ks)
    per_user: dict[str, list[float]] = {f"{m}@{k}": [] for m in METRICS for k in ks}
    recommended_at_k: dict[int, set[int]] = {k: set() for k in ks}
    n_users = 0
    for user_id, rel in relevant.items():
        ctx = feats.context_for_user(user_id)
        if ctx.is_cold or not rel:
            continue
        ranked = [r.item_id for r in model.recommend(ctx, kmax, exclude_seen=True, explain=False)]
        n_users += 1
        for k in ks:
            recommended_at_k[k].update(ranked[:k])
            for name, fn in METRICS.items():
                per_user[f"{name}@{k}"].append(fn(ranked, rel, k))
    metrics = {key: (float(np.mean(v)) if v else 0.0) for key, v in per_user.items()}
    # standard error of the mean over users: with ~1 relevant item per user these metrics are
    # noisy, and differences smaller than ~2 SE should not be read as real
    stderr = {key: (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0) for key, v in per_user.items()}
    for k in ks:
        metrics[f"coverage@{k}"] = len(recommended_at_k[k]) / feats.n_items
    return {
        "n_users": n_users,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "stderr": {k: round(v, 4) for k, v in stderr.items()},
    }


def _weight_grid(step: float, max_popularity: float | None = None) -> list[dict[str, float]]:
    units = int(round(1 / step))
    grid = []
    for combo in itertools.product(range(units + 1), repeat=len(COMPONENT_NAMES)):
        if sum(combo) == units:
            w = {n: c / units for n, c in zip(COMPONENT_NAMES, combo)}
            if max_popularity is None or w["popularity"] <= max_popularity + 1e-9:
                grid.append(w)
    return grid


def tune_hybrid_weights(
    bundle: DataBundle,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    k: int = 10,
    step: float = 0.25,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    objective: str = "ndcg",
    max_popularity: float | None = 0.3,
) -> tuple[dict[str, float], list[dict]]:
    """Grid-search blend weights on the VALIDATION split (models fitted on TRAIN only).

    `max_popularity` is a product constraint, not a statistical one: with a small validation
    split an unconstrained search happily picks a popularity-heavy blend whose edge over a
    personalised one is within noise, which would make the "hybrid" mostly a trending list.
    Pass None to search the full simplex.
    """
    train, val, _ = temporal_split(bundle.interactions, val_frac, test_frac)
    feats = build_features(bundle.users, bundle.items, train, half_life_days=half_life_days)
    models = fit_models(build_models(), feats)
    rel = relevant_map(val)
    hybrid = models["hybrid"]
    trials = []
    for w in _weight_grid(step, max_popularity):
        hybrid.set_weights(w)
        res = evaluate_model(hybrid, feats, rel, [k])
        trials.append({"weights": w, "n_users": res["n_users"], **res["metrics"]})
    best = max(trials, key=lambda t: t[f"{objective}@{k}"])
    return best["weights"], trials


def run_comparison(
    bundle: DataBundle,
    ks: Iterable[int] = (5, 10, 20),
    hybrid_weights: dict[str, float] | None = None,
    model_names: Iterable[str] | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    include_random: bool = True,
) -> dict:
    """Fit on TRAIN+VAL, score on TEST, for every requested model."""
    train, val, test = temporal_split(bundle.interactions, val_frac, test_frac)
    fit_df = pd.concat([train, val])
    feats = build_features(bundle.users, bundle.items, fit_df, half_life_days=half_life_days)
    models = fit_models(build_models(hybrid_weights), feats)
    names = list(model_names) if model_names else list(models)
    unknown = [n for n in names if n not in models]
    if unknown:
        raise ValueError(f"unknown models: {unknown}; available: {list(models)}")
    rel = relevant_map(test)
    results = {}
    if include_random:
        rnd = RandomRecommender().fit(feats)
        results["random"] = evaluate_model(rnd, feats, rel, ks)
    for name in names:
        results[name] = evaluate_model(models[name], feats, rel, ks)
    return {
        "protocol": {
            "split": "per-user temporal leave-last-out",
            "val_frac": val_frac,
            "test_frac": test_frac,
            "n_train_interactions": int(len(train)),
            "n_val_interactions": int(len(val)),
            "n_test_interactions": int(len(test)),
            "half_life_days": half_life_days,
            "hybrid_weights": models["hybrid"].weights,
        },
        "ks": sorted(set(int(k) for k in ks)),
        "results": results,
    }


def format_markdown_table(report: dict, k: int) -> str:
    cols = [f"precision@{k}", f"recall@{k}", f"map@{k}", f"ndcg@{k}", f"coverage@{k}"]
    lines = ["| model | users | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, res in report["results"].items():
        se = res.get("stderr", {})
        vals = " | ".join(
            f"{res['metrics'][c]:.4f}" + (f" ±{se[c]:.4f}" if c in se and not c.startswith("coverage") else "")
            for c in cols
        )
        lines.append(f"| {name} | {res['n_users']} | {vals} |")
    return "\n".join(lines)

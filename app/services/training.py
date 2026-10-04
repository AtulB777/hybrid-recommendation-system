"""OFFLINE side: fit final models and precompute recommendations into the database.

Nothing here runs inside the API process. `training/train.py` orchestrates these steps as a
batch job (cron / Airflow / docker-compose `trainer` service).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.engine import Engine

from app.features.engineering import DEFAULT_HALF_LIFE_DAYS, build_features
from app.models.orm import Recommendation
from app.recommenders.registry import build_models, fit_models
from app.services.artifacts import ModelBundle, make_version
from app.services.ingestion import DataBundle


def fit_final_bundle(
    data: DataBundle,
    hybrid_weights: dict[str, float] | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    offline_metrics: dict | None = None,
) -> ModelBundle:
    """Fit on ALL interactions (the production model; evaluation used held-out splits)."""
    feats = build_features(data.users, data.items, data.interactions, half_life_days=half_life_days)
    models = fit_models(build_models(hybrid_weights), feats)
    weights = models["hybrid"].weights
    return ModelBundle(
        version=make_version(feats, weights),
        trained_at=datetime.now(timezone.utc),
        feats=feats,
        models=models,
        hybrid_weights=weights,
        config={"half_life_days": half_life_days, "models": {n: m.params() for n, m in models.items()}},
        offline_metrics=offline_metrics,
    )


def generate_batch_recommendations(
    bundle: ModelBundle, engine: Engine, k: int = 20, model_name: str = "hybrid"
) -> int:
    """Precompute top-K (with explanations) for every user and store them in `recommendations`.

    Users with no history get the popularity-driven cold-start list. Rows from older model
    versions are replaced so the table never mixes versions.
    """
    model = bundle.models[model_name]
    feats = bundle.feats
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for user_id in feats.user_ids:
        ctx = feats.context_for_user(int(user_id))
        for rec in model.recommend(ctx, k, exclude_seen=True, explain=True):
            rows.append(
                {
                    "user_id": int(user_id), "item_id": rec.item_id, "model_name": model_name,
                    "model_version": bundle.version, "rank": rec.rank, "score": rec.score,
                    "reason_type": rec.reason_type, "explanation": rec.explanation, "generated_at": now,
                }
            )
    df = pd.DataFrame(rows)
    with engine.begin() as conn:
        conn.execute(delete(Recommendation).where(Recommendation.model_name == model_name))
        if len(df):
            df.to_sql("recommendations", conn, if_exists="append", index=False, chunksize=500)
    return len(df)

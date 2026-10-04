"""ONLINE side: low-latency recommendation serving.

Request path for GET /recommendations/{user_id}
  cache  ->  precomputed rows from the offline job  ->  real-time scoring with the loaded model

A precomputed list is only used if (a) it was produced by the *currently loaded* model version
and (b) the user has no interactions newer than the model's training data. Otherwise the user
is scored in real time from their live DB history, so fresh behaviour is never ignored.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.session import make_session_factory
from app.evaluation.runner import run_comparison
from app.models.orm import Interaction, Item, Recommendation, User
from app.services.artifacts import ModelBundle
from app.services.cache import CacheBackend
from app.services.ingestion import load_bundle_from_db


class ModelNotReady(RuntimeError):
    pass


class NotFound(LookupError):
    pass


class RecommendationService:
    def __init__(
        self, engine: Engine, cache: CacheBackend, bundle: ModelBundle | None,
        cache_ttl: int = 300, default_k: int = 10,
    ) -> None:
        self.engine = engine
        self.sessions: sessionmaker[Session] = make_session_factory(engine)
        self.cache, self.bundle = cache, bundle
        self.cache_ttl, self.default_k = cache_ttl, default_k

    # ------------------------------------------------------------------ helpers
    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def _require_bundle(self) -> ModelBundle:
        if self.bundle is None:
            raise ModelNotReady("No trained model loaded. Run `python -m training.train` first.")
        return self.bundle

    def _item_payload(self, item_idx: int) -> dict[str, Any]:
        row = self._require_bundle().feats.items.iloc[item_idx]
        return {"item_id": int(row.name), "title": str(row["title"]), "genres": list(row["genres_list"])}

    def _assert_user(self, session: Session, user_id: int) -> None:
        if session.get(User, user_id) is None:
            raise NotFound(f"user {user_id} not found")

    def _history_df(self, session: Session, user_id: int) -> pd.DataFrame:
        rows = session.execute(
            select(Interaction.item_id, Interaction.event_type, Interaction.weight, Interaction.timestamp)
            .where(Interaction.user_id == user_id)
        ).all()
        return pd.DataFrame(rows, columns=["item_id", "event_type", "weight", "timestamp"])

    def _model(self, name: str):
        bundle = self._require_bundle()
        if name not in bundle.models:
            raise ValueError(f"unknown model '{name}'. Available: {list(bundle.models)}")
        return bundle.models[name]

    def _has_fresh_interactions(self, session: Session, user_id: int) -> bool:
        latest = session.execute(
            select(func.max(Interaction.timestamp)).where(Interaction.user_id == user_id)
        ).scalar()
        return latest is not None and pd.Timestamp(latest) > self._require_bundle().data_as_of

    def _response(self, user_id, model, k, source, recs: list[dict]) -> dict[str, Any]:
        b = self._require_bundle()
        return {
            "user_id": user_id, "model": model, "k": k, "source": source,
            "model_version": b.version, "items": recs,
        }

    # ---------------------------------------------------------------- real-time
    def _realtime(
        self, session: Session, user_id: int | None, history_item_ids: list[int], model_name: str,
        k: int, exclude_seen: bool, exclude_item_ids: list[int],
    ) -> list[dict]:
        b = self._require_bundle()
        history = self._history_df(session, user_id) if user_id is not None else None
        ctx = b.feats.context_from_history(history, user_id, history_item_ids)
        exclude_idx = [b.feats.item_index[i] for i in exclude_item_ids if i in b.feats.item_index]
        import numpy as np

        recs = self._model(model_name).recommend(
            ctx, k, exclude_seen=exclude_seen, explain=True, exclude_idx=np.array(exclude_idx, dtype=np.int64)
        )
        return [
            {**self._item_payload(r.item_idx), "rank": r.rank, "score": round(r.score, 6),
             "explanation": r.explanation, "reason_type": r.reason_type}
            for r in recs
        ]

    # ------------------------------------------------------------------ public
    def get_recommendations(self, user_id: int, k: int, model: str = "hybrid", exclude_seen: bool = True) -> dict:
        b = self._require_bundle()
        self._model(model)  # validates the name
        key = f"recs:{b.version}:{user_id}:{model}:{k}:{int(exclude_seen)}"
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "source": "cache"}

        with self.sessions() as session:
            self._assert_user(session, user_id)
            recs, source = None, "realtime"
            if model == "hybrid" and exclude_seen and not self._has_fresh_interactions(session, user_id):
                rows = session.execute(
                    select(Recommendation)
                    .where(Recommendation.user_id == user_id, Recommendation.model_name == "hybrid",
                           Recommendation.model_version == b.version)
                    .order_by(Recommendation.rank).limit(k)
                ).scalars().all()
                if len(rows) >= k:
                    recs = [
                        {**self._item_payload(b.feats.item_index[r.item_id]), "rank": r.rank,
                         "score": round(r.score, 6), "explanation": r.explanation, "reason_type": r.reason_type}
                        for r in rows
                    ]
                    source = "precomputed"
            if recs is None:
                recs = self._realtime(session, user_id, [], model, k, exclude_seen, [])
        payload = self._response(user_id, model, k, source, recs)
        self.cache.set(key, payload, self.cache_ttl)
        return payload

    def recommend(
        self, user_id: int | None, history_item_ids: list[int], k: int, model: str,
        exclude_seen: bool, exclude_item_ids: list[int],
    ) -> dict:
        """POST /recommendations: always real-time. Supports anonymous / session-based requests
        (no user_id, just recently viewed items) which is the live cold-start path."""
        self._model(model)
        with self.sessions() as session:
            if user_id is not None:
                self._assert_user(session, user_id)
            recs = self._realtime(session, user_id, history_item_ids, model, k, exclude_seen, exclude_item_ids)
        return self._response(user_id, model, k, "realtime", recs)

    def similar_items(self, item_id: int, k: int, method: str = "hybrid") -> dict:
        b = self._require_bundle()
        idx = b.feats.item_index.get(item_id)
        if idx is None:
            raise NotFound(f"item {item_id} not found")
        pairs = b.models["hybrid"].similar_items(idx, k, method)
        return {
            "item_id": item_id, "method": method,
            "is_cold_item": bool(b.feats.items.iloc[idx]["is_cold"]),
            "items": [{**self._item_payload(j), "similarity": round(s, 6)} for j, s in pairs],
        }

    def user_history(self, user_id: int, limit: int = 50) -> dict:
        with self.sessions() as session:
            self._assert_user(session, user_id)
            rows = session.execute(
                select(Interaction, Item.title, Item.genres)
                .join(Item, Item.item_id == Interaction.item_id)
                .where(Interaction.user_id == user_id)
                .order_by(Interaction.timestamp.desc()).limit(limit)
            ).all()
            total = session.execute(
                select(func.count()).select_from(Interaction).where(Interaction.user_id == user_id)
            ).scalar_one()
        genre_counts: dict[str, int] = {}
        interactions = []
        for inter, title, genres in rows:
            for g in genres.split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1
            interactions.append({
                "item_id": inter.item_id, "title": title, "genres": genres.split("|"),
                "event_type": inter.event_type, "weight": inter.weight, "timestamp": inter.timestamp,
            })
        top = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]
        return {"user_id": user_id, "total_interactions": total, "top_genres": top, "interactions": interactions}

    def list_models(self) -> dict:
        b = self._require_bundle() if self.ready else None
        if b is None:
            return {"ready": False, "model_version": None, "trained_at": None, "models": []}
        models = [
            {"name": n, "description": m.description, "params": m.params(), "fitted": m.feats is not None}
            for n, m in b.models.items()
        ]
        return {
            "ready": True, "model_version": b.version, "trained_at": b.trained_at,
            "data_as_of": b.data_as_of.to_pydatetime(), "n_users": b.feats.n_users,
            "n_items": b.feats.n_items, "hybrid_weights": b.hybrid_weights,
            "offline_metrics": b.offline_metrics, "models": models,
        }

    def evaluate(self, k_values: list[int], models: list[str] | None, test_frac: float) -> dict:
        """POST /evaluate. Re-runs the offline protocol on the DB's data. Synchronous and
        CPU-bound: fine for this dataset, but at scale it should be a queued background job."""
        b = self._require_bundle()
        key = f"eval:{b.version}:{sorted(k_values)}:{sorted(models or [])}:{test_frac}"
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        data = load_bundle_from_db(self.engine)
        report = run_comparison(
            data, ks=k_values, hybrid_weights=b.hybrid_weights, model_names=models,
            half_life_days=b.feats.half_life_days, test_frac=test_frac,
        )
        report["cached"] = False
        self.cache.set(key, report, self.cache_ttl * 12)
        return report

"""Model artifact = the hand-off between OFFLINE training and ONLINE serving.

Security note: artifacts are joblib (pickle) files. Only load artifacts your own training job
produced; never load one from an untrusted source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.features.engineering import Features
from app.recommenders.base import BaseRecommender


@dataclass
class ModelBundle:
    version: str
    trained_at: datetime
    feats: Features
    models: dict[str, BaseRecommender]
    hybrid_weights: dict[str, float]
    config: dict = field(default_factory=dict)
    offline_metrics: dict | None = None  # test-split comparison from training time

    @property
    def data_as_of(self) -> pd.Timestamp:
        return self.feats.as_of


def make_version(feats: Features, weights: dict[str, float]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha1(
        f"{feats.n_users}|{feats.n_items}|{feats.matrix.nnz}|{sorted(weights.items())}".encode()
    ).hexdigest()[:8]
    return f"{stamp}-{digest}"


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)  # atomic swap: the API never sees a half-written file


def load_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)

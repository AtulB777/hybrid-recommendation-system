from __future__ import annotations

import numpy as np

from app.features.engineering import Features, UserContext
from app.recommenders.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Sanity floor for evaluation only (never served): any real model must beat this."""

    name = "random"
    reason_type = "random"
    description = "Uniform random ranking. Evaluation floor, not exposed by the API."

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def fit(self, feats: Features) -> "RandomRecommender":
        super().fit(feats)
        self._rng = np.random.default_rng(self.seed)
        return self

    def score(self, ctx: UserContext) -> np.ndarray:
        return self._rng.random(self.feats.n_items)

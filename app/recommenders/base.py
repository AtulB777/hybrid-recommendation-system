"""Common interface for all recommenders.

Contract: a model scores *every* catalog item for a `UserContext`. Candidate selection,
seen-item filtering and top-K ranking happen in `recommend`, so all models share them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.features.engineering import Features, UserContext


@dataclass
class ScoredItem:
    item_id: int
    item_idx: int
    score: float
    rank: int
    explanation: str = ""
    reason_type: str = ""


def top_k_indices(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest finite scores, best first. Ties break toward lower index."""
    s = np.asarray(scores, dtype=np.float64).copy()
    s[~np.isfinite(s)] = -np.inf
    if exclude is not None and len(exclude):
        s[exclude] = -np.inf
    n_valid = int(np.isfinite(s).sum())
    k = min(k, n_valid)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
    order = np.lexsort((part, -s[part]))
    return part[order]


def format_support(titles: list[str]) -> str:
    quoted = [f'"{t}"' for t in titles]
    if not quoted:
        return "Recommended based on your recent activity."
    joined = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    return f"Recommended because you interacted with {joined}."


class BaseRecommender(ABC):
    name = "base"
    reason_type = "generic"
    description = ""

    def __init__(self) -> None:
        self.feats: Features | None = None

    # ---- training ----
    def fit(self, feats: Features) -> "BaseRecommender":
        self.feats = feats
        return self

    def params(self) -> dict:
        return {}

    # ---- scoring ----
    @abstractmethod
    def score(self, ctx: UserContext) -> np.ndarray:
        """Return a float array of length n_items (higher = better)."""

    def supporting_items(self, ctx: UserContext, item_idx: int, n: int = 2) -> list[int]:
        """Indices of the user's past items that best justify recommending `item_idx`."""
        return []

    def _titles(self, idxs: list[int]) -> list[str]:
        assert self.feats is not None
        return [str(self.feats.items["title"].iloc[i]) for i in idxs]

    def explain(self, ctx: UserContext, item_idx: int) -> str:
        return format_support(self._titles(self.supporting_items(ctx, item_idx)))

    # hooks so composite models (hybrid) can compute scores + explanation state once
    def _score_with_state(self, ctx: UserContext, exclude_seen: bool):
        return self.score(ctx), None

    def _explain_item(self, ctx: UserContext, item_idx: int, state) -> tuple[str, str]:
        return self.explain(ctx, item_idx), self.reason_type

    # ---- ranking ----
    def recommend(
        self, ctx: UserContext, k: int = 10, exclude_seen: bool = True, explain: bool = True,
        exclude_idx: np.ndarray | None = None,
    ) -> list[ScoredItem]:
        assert self.feats is not None, "model is not fitted"
        scores, state = self._score_with_state(ctx, exclude_seen)
        excl = ctx.seen_idx if exclude_seen else np.empty(0, dtype=np.int64)
        if exclude_idx is not None and len(exclude_idx):
            excl = np.union1d(excl, exclude_idx)
        top = top_k_indices(scores, k, excl)
        out = []
        for rank, j in enumerate(top, start=1):
            text, rtype = self._explain_item(ctx, int(j), state) if explain else ("", self.reason_type)
            out.append(ScoredItem(int(self.feats.item_ids[j]), int(j), float(scores[j]), rank, text, rtype))
        return out

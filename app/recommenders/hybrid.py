from __future__ import annotations

import numpy as np

from app.features.engineering import Features, UserContext
from app.recommenders.base import BaseRecommender, top_k_indices

DEFAULT_WEIGHTS = {"popularity": 0.10, "content": 0.30, "cf_item": 0.40, "cf_user": 0.20}


class HybridRecommender(BaseRecommender):
    """Weighted hybrid with explicit candidate generation and score normalisation.

    Pipeline per request
    1. Candidate generation: union of each component's top `candidate_pool` unseen items.
    2. Normalisation: every component's scores are min-max scaled over the candidate set so
       cosine scores, vote sums and popularity share one 0-1 scale.
    3. Ranking: final = sum_c  w_c(history) * norm_c.
    4. Explanation: the component contributing most to an item's score explains it.

    Cold-start handling lives in `effective_weights`: with no history the model is pure
    popularity; with a short history (< `min_history` items) the collaborative weights are
    scaled down proportionally and the freed mass goes to content + popularity, because
    similarity estimates from 1-2 interactions are noisy.
    """

    name = "hybrid"
    reason_type = "hybrid"
    description = "Weighted blend of popularity, content, item-CF and user-CF scores over a unioned candidate pool, with history-length-aware weights."

    def __init__(
        self,
        components: dict[str, BaseRecommender],
        weights: dict[str, float] | None = None,
        candidate_pool: int = 100,
        min_history: int = 3,
    ) -> None:
        super().__init__()
        self.components = components
        self.weights = self._normalise_weights(weights or DEFAULT_WEIGHTS, components)
        self.candidate_pool = candidate_pool
        self.min_history = min_history

    @staticmethod
    def _normalise_weights(weights: dict[str, float], components) -> dict[str, float]:
        w = {k: float(weights.get(k, 0.0)) for k in components}
        if any(v < 0 for v in w.values()):
            raise ValueError("weights must be non-negative")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        return {k: v / total for k, v in w.items()}

    def set_weights(self, weights: dict[str, float]) -> None:
        self.weights = self._normalise_weights(weights, self.components)

    def params(self) -> dict:
        return {"weights": self.weights, "candidate_pool": self.candidate_pool, "min_history": self.min_history}

    def fit(self, feats: Features) -> "HybridRecommender":
        # components are fitted by the caller (they are shared objects); this only binds features
        self.feats = feats
        return self

    # ---- weighting policy ----
    def effective_weights(self, n_history: int) -> dict[str, float]:
        if n_history <= 0:
            return {k: (1.0 if k == "popularity" else 0.0) for k in self.components}
        w = dict(self.weights)
        if n_history < self.min_history:
            scale = n_history / self.min_history
            freed = 0.0
            for k in ("cf_item", "cf_user"):
                if k in w:
                    freed += w[k] * (1 - scale)
                    w[k] *= scale
            for k in ("content", "popularity"):
                if k in w:
                    w[k] += freed / 2
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ---- scoring ----
    def _breakdown(self, ctx: UserContext, exclude_seen: bool = True):
        w = self.effective_weights(ctx.n_history)
        active = {k: self.components[k] for k, v in w.items() if v > 0}
        raw = {k: m.score(ctx) for k, m in active.items()}
        excl = ctx.seen_idx if exclude_seen else None
        cand: set[int] = set()
        for s in raw.values():
            cand.update(top_k_indices(s, self.candidate_pool, excl).tolist())
        n = self.feats.n_items
        final = np.full(n, -np.inf)
        contrib = {k: np.zeros(n) for k in raw}
        if cand:
            idx = np.fromiter(cand, dtype=np.int64)
            total = np.zeros(len(idx))
            for k, s in raw.items():
                v = s[idx].astype(np.float64)
                lo, hi = v.min(), v.max()
                norm = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
                c = w[k] * norm
                contrib[k][idx] = c
                total += c
            final[idx] = total
        return final, contrib

    def score(self, ctx: UserContext) -> np.ndarray:
        return self._breakdown(ctx)[0]

    def _score_with_state(self, ctx: UserContext, exclude_seen: bool):
        final, contrib = self._breakdown(ctx, exclude_seen)
        return final, contrib

    PERSONALISED_SHARE = 0.25

    def _explain_item(self, ctx: UserContext, item_idx: int, state) -> tuple[str, str]:
        """Explain with the strongest *personalised* component when it supplies at least
        PERSONALISED_SHARE of the item's blended score; otherwise say it is popular. Popularity
        usually has the single largest term (it is dense), so "argmax component" alone would
        label nearly everything as trending even when a real personal signal exists."""
        contrib = state or {}
        total = sum(c[item_idx] for c in contrib.values())
        personal = {k: c[item_idx] for k, c in contrib.items() if k != "popularity"}
        if total > 0 and personal:
            for name in sorted(personal, key=personal.get, reverse=True):
                if personal[name] / total < self.PERSONALISED_SHARE:
                    break
                model = self.components[name]
                if model.supporting_items(ctx, item_idx, 1):
                    return model.explain(ctx, item_idx), model.reason_type
        return self.components["popularity"].explain(ctx, item_idx), "popularity"

    # ---- item-to-item ----
    def similar_items(self, item_idx: int, k: int = 10, method: str = "hybrid") -> list[tuple[int, float]]:
        content = self.components["content"].item_similarity_row(item_idx).astype(np.float64)
        cf = self.components["cf_item"].item_similarity_row(item_idx).astype(np.float64)
        if method == "content":
            sims = content
        elif method == "collaborative":
            sims = cf
        elif method == "hybrid":
            # items with no interaction history have an all-zero CF row -> content only
            sims = 0.5 * content + 0.5 * cf if cf.max() > 0 else content
        else:
            raise ValueError("method must be one of: hybrid, content, collaborative")
        sims = sims.copy()
        sims[item_idx] = -np.inf
        order = top_k_indices(sims, k)
        return [(int(j), float(sims[j])) for j in order if sims[j] > 0]

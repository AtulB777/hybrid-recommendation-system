from __future__ import annotations

import numpy as np

from app.features.engineering import Features, UserContext
from app.recommenders.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Non-personalised baseline: items ranked by time-decayed, event-weighted interaction mass.

    This is also the cold-start fallback for users with no history.
    """

    name = "popularity"
    reason_type = "popularity"
    description = "Ranks items by recency-weighted interaction volume. Same list for everyone; the cold-start fallback."

    def fit(self, feats: Features) -> "PopularityRecommender":
        super().fit(feats)
        self.pop = feats.items["popularity_score"].to_numpy(dtype=np.float64)
        top = self.pop.max() if len(self.pop) else 0.0
        self.pop = self.pop / top if top > 0 else self.pop
        return self

    def score(self, ctx: UserContext) -> np.ndarray:
        return self.pop

    def explain(self, ctx: UserContext, item_idx: int) -> str:
        return "Recommended because it is popular with other users."

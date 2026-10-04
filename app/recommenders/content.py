from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.features.engineering import Features, UserContext
from app.recommenders.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """TF-IDF item vectors + cosine similarity to a weighted user profile.

    user_profile = L2-normalised sum of TF-IDF vectors of the items the user interacted with,
    weighted by (event strength x recency decay). Score(item) = cosine(profile, item).
    Needs no interaction data about the *candidate* item, so it works for brand-new items.
    """

    name = "content"
    reason_type = "content"
    description = "TF-IDF over title/genres/description; cosine similarity between item vectors and the user's weighted profile."

    def __init__(self, ngram_range=(1, 2), min_df: int = 2, max_features: int | None = 20000) -> None:
        super().__init__()
        self.ngram_range, self.min_df, self.max_features = ngram_range, min_df, max_features

    def params(self) -> dict:
        return {"ngram_range": list(self.ngram_range), "min_df": self.min_df, "max_features": self.max_features}

    def fit(self, feats: Features) -> "ContentBasedRecommender":
        super().fit(feats)
        self.vectorizer = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=self.ngram_range,
            min_df=self.min_df, max_features=self.max_features,
        )
        self.X = self.vectorizer.fit_transform(feats.items["text"]).tocsr()  # rows are L2-normalised
        return self

    def score(self, ctx: UserContext) -> np.ndarray:
        if ctx.is_cold:
            return np.zeros(self.feats.n_items)
        profile = normalize(ctx.vector @ self.X)  # (1, V)
        return np.asarray((self.X @ profile.T).todense()).ravel()

    def supporting_items(self, ctx: UserContext, item_idx: int, n: int = 2) -> list[int]:
        seen = ctx.seen_idx
        if not len(seen):
            return []
        sims = np.asarray((self.X[seen] @ self.X[item_idx].T).todense()).ravel()
        contrib = sims * ctx.vector.data
        order = np.argsort(-contrib, kind="stable")[:n]
        return [int(seen[i]) for i in order if contrib[i] > 0]

    def similar_items(self, item_idx: int, k: int = 10) -> list[tuple[int, float]]:
        sims = np.asarray((self.X @ self.X[item_idx].T).todense()).ravel()
        sims[item_idx] = -np.inf
        order = np.argsort(-sims, kind="stable")[:k]
        return [(int(j), float(sims[j])) for j in order if np.isfinite(sims[j])]

    def item_similarity_row(self, item_idx: int) -> np.ndarray:
        return np.asarray((self.X @ self.X[item_idx].T).todense()).ravel()

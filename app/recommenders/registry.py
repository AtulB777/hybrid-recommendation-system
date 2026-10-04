from __future__ import annotations

from app.features.engineering import Features
from app.recommenders.base import BaseRecommender
from app.recommenders.collaborative import ItemKNNRecommender, UserKNNRecommender
from app.recommenders.content import ContentBasedRecommender
from app.recommenders.hybrid import HybridRecommender
from app.recommenders.popularity import PopularityRecommender

COMPONENT_NAMES = ("popularity", "content", "cf_item", "cf_user")


def build_models(weights: dict[str, float] | None = None) -> dict[str, BaseRecommender]:
    components = {
        "popularity": PopularityRecommender(),
        "content": ContentBasedRecommender(),
        "cf_item": ItemKNNRecommender(),
        "cf_user": UserKNNRecommender(),
    }
    return {**components, "hybrid": HybridRecommender(components, weights)}


def fit_models(models: dict[str, BaseRecommender], feats: Features) -> dict[str, BaseRecommender]:
    for name, model in models.items():
        if name != "hybrid":
            model.fit(feats)
    models["hybrid"].fit(feats)  # after components so it can bind features
    return models

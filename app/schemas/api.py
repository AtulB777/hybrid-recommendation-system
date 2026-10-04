from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    rank: int
    item_id: int
    title: str
    genres: list[str]
    score: float
    explanation: str = Field(description='e.g. Recommended because you interacted with "A" and "B".')
    reason_type: str = Field(description="popularity | content | collaborative")


class RecommendationResponse(BaseModel):
    user_id: int | None
    model: str
    k: int
    source: Literal["cache", "precomputed", "realtime"]
    model_version: str
    items: list[RecommendedItem]


class RecommendationRequest(BaseModel):
    user_id: int | None = Field(None, description="Known user. Their live DB history is used.")
    history_item_ids: list[int] = Field(
        default_factory=list,
        description="Extra/anonymous history, e.g. items viewed this session. Enables cold-start requests.",
        max_length=200,
    )
    k: int = Field(10, ge=1, le=100)
    model: str = "hybrid"
    exclude_seen: bool = True
    exclude_item_ids: list[int] = Field(default_factory=list, max_length=500)


class SimilarItem(BaseModel):
    item_id: int
    title: str
    genres: list[str]
    similarity: float


class SimilarItemsResponse(BaseModel):
    item_id: int
    method: str
    is_cold_item: bool
    items: list[SimilarItem]


class HistoryEntry(BaseModel):
    item_id: int
    title: str
    genres: list[str]
    event_type: str
    weight: float
    timestamp: datetime


class HistoryResponse(BaseModel):
    user_id: int
    total_interactions: int
    top_genres: list[str]
    interactions: list[HistoryEntry]


class ModelInfo(BaseModel):
    name: str
    description: str
    params: dict[str, Any]
    fitted: bool


class ModelsResponse(BaseModel):
    ready: bool
    model_version: str | None
    trained_at: datetime | None
    data_as_of: datetime | None = None
    n_users: int | None = None
    n_items: int | None = None
    hybrid_weights: dict[str, float] | None = None
    offline_metrics: dict[str, Any] | None = None
    models: list[ModelInfo]


class EvaluateRequest(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    models: list[str] | None = Field(None, description="Subset of models; default all.")
    test_frac: float = Field(0.2, ge=0.05, le=0.5)

    def validated_ks(self) -> list[int]:
        if not self.k_values or any(k < 1 or k > 100 for k in self.k_values):
            raise ValueError("k_values must be non-empty with each value in 1..100")
        return self.k_values


class EvaluateResponse(BaseModel):
    protocol: dict[str, Any]
    ks: list[int]
    results: dict[str, Any]
    cached: bool = False

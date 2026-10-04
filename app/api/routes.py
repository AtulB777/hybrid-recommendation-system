from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.schemas.api import (
    EvaluateRequest, EvaluateResponse, HistoryResponse, ModelsResponse,
    RecommendationRequest, RecommendationResponse, SimilarItemsResponse,
)
from app.services.serving import RecommendationService

router = APIRouter()


def get_service(request: Request) -> RecommendationService:
    return request.app.state.service


@router.get("/health", tags=["meta"])
def health(service: RecommendationService = Depends(get_service)) -> dict:
    return {"status": "ok", "model_loaded": service.ready}


@router.get("/recommendations/{user_id}", response_model=RecommendationResponse, tags=["recommendations"])
def get_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=100),
    model: str = "hybrid",
    exclude_seen: bool = True,
    service: RecommendationService = Depends(get_service),
):
    """Top-K for a user: cache -> precomputed (offline job) -> real-time fallback."""
    return service.get_recommendations(user_id, k, model, exclude_seen)


@router.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
def post_recommendations(body: RecommendationRequest, service: RecommendationService = Depends(get_service)):
    """Real-time scoring. Accepts a user_id and/or ad-hoc history (anonymous cold-start)."""
    return service.recommend(
        body.user_id, body.history_item_ids, body.k, body.model, body.exclude_seen, body.exclude_item_ids
    )


@router.get("/items/{item_id}/similar", response_model=SimilarItemsResponse, tags=["items"])
def similar_items(
    item_id: int,
    k: int = Query(10, ge=1, le=100),
    method: Literal["hybrid", "content", "collaborative"] = "hybrid",
    service: RecommendationService = Depends(get_service),
):
    return service.similar_items(item_id, k, method)


@router.get("/users/{user_id}/history", response_model=HistoryResponse, tags=["users"])
def user_history(
    user_id: int, limit: int = Query(50, ge=1, le=500), service: RecommendationService = Depends(get_service)
):
    return service.user_history(user_id, limit)


@router.get("/models", response_model=ModelsResponse, tags=["models"])
def list_models(service: RecommendationService = Depends(get_service)):
    return service.list_models()


@router.post("/evaluate", response_model=EvaluateResponse, tags=["models"])
def evaluate(body: EvaluateRequest, service: RecommendationService = Depends(get_service)):
    """Re-run the offline comparison (P/R/MAP/NDCG@K) on the database's data."""
    try:
        ks = body.validated_ks()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return service.evaluate(ks, body.models, body.test_frac)

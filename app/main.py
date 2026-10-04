"""FastAPI application factory (ONLINE serving process)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.database.session import init_db, make_engine
from app.services.artifacts import load_bundle
from app.services.cache import make_cache
from app.services.serving import ModelNotReady, NotFound, RecommendationService

log = logging.getLogger("recsys")


def create_app(settings: Settings | None = None, service: RecommendationService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
        else:
            engine = make_engine(settings.database_url)
            init_db(engine)
            bundle = None
            if settings.artifact_path.exists():
                bundle = load_bundle(settings.artifact_path)
                log.info("loaded model %s", bundle.version)
            else:
                log.warning("no model artifact at %s - run `python -m training.train`", settings.artifact_path)
            app.state.service = RecommendationService(
                engine, make_cache(settings.cache_backend, settings.redis_url), bundle,
                settings.cache_ttl_seconds, settings.default_k,
            )
        yield

    app = FastAPI(
        title="Hybrid Recommendation Engine",
        version="1.0.0",
        description="Popularity + content-based + collaborative filtering, blended by a hybrid ranker.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ModelNotReady)
    async def _not_ready(_: Request, exc: ModelNotReady):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()

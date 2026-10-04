"""Central configuration. Every value can be overridden with a RECSYS_* env var."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # SQLite keeps `make quickstart` dependency-free; docker-compose points this at PostgreSQL.
    database_url: str = f"sqlite:///{ROOT / 'data' / 'recsys.db'}"
    raw_data_dir: Path = ROOT / "data" / "raw"
    artifact_path: Path = ROOT / "artifacts" / "model_bundle.joblib"
    docs_dir: Path = ROOT / "docs"

    # Serving
    default_k: int = 10
    batch_k: int = 20  # how many recommendations the offline job precomputes per user
    cache_backend: str = "memory"  # memory | null | redis
    cache_ttl_seconds: int = 300
    redis_url: str = "redis://localhost:6379/0"

    # Modelling
    half_life_days: float = 120.0

    model_config = SettingsConfigDict(env_prefix="RECSYS_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

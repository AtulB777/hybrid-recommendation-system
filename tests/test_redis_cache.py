"""Runs only when a Redis is reachable: RECSYS_TEST_REDIS_URL=redis://localhost:6379/15 pytest"""
import os

import pytest

URL = os.environ.get("RECSYS_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set RECSYS_TEST_REDIS_URL to run against a live Redis")


def test_redis_roundtrip_ttl_and_delete():
    import time
    from app.services.cache import make_cache

    c = make_cache("redis", URL)
    c.clear()
    c.set("k", {"a": [1, 2]}, ttl_seconds=1)
    assert c.get("k") == {"a": [1, 2]}
    time.sleep(1.2)
    assert c.get("k") is None
    c.set("k2", {"v": 1}, 60); c.delete("k2")
    assert c.get("k2") is None


def test_service_runs_on_redis_cache(engine, model_bundle, warm_user):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.services.cache import make_cache
    from app.services.serving import RecommendationService

    cache = make_cache("redis", URL); cache.clear()
    with TestClient(create_app(service=RecommendationService(engine, cache, model_bundle))) as c:
        assert c.get(f"/recommendations/{warm_user}?k=5").json()["source"] == "precomputed"
        assert c.get(f"/recommendations/{warm_user}?k=5").json()["source"] == "cache"

"""Cache abstraction. The serving layer only talks to `CacheBackend`, so moving from the
in-process cache to Redis is a config change (RECSYS_CACHE_BACKEND=redis), not a code change.

Values must be JSON-serialisable dicts: that is what makes the Redis backend a drop-in.
RedisCache is covered by tests/test_redis_cache.py, which is skipped unless
RECSYS_TEST_REDIS_URL points at a live Redis (so the default `pytest` run needs no services).
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class NullCache(CacheBackend):
    def get(self, key): return None
    def set(self, key, value, ttl_seconds): pass
    def delete(self, key): pass
    def clear(self): pass


class InMemoryTTLCache(CacheBackend):
    def __init__(self, max_entries: int = 10_000, clock=time.monotonic) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._max, self._clock = max_entries, clock
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry[0] <= self._clock():
                self._data.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(entry[1])  # round-trip proves values are Redis-compatible

    def set(self, key, value, ttl_seconds):
        with self._lock:
            if len(self._data) >= self._max:
                self._data.pop(next(iter(self._data)))  # evict oldest insertion
            self._data[key] = (self._clock() + ttl_seconds, json.dumps(value))

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)

    def clear(self):
        with self._lock:
            self._data.clear()


class RedisCache(CacheBackend):
    def __init__(self, url: str) -> None:
        try:
            import redis  # optional dependency: pip install redis
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("RECSYS_CACHE_BACKEND=redis requires `pip install redis`") from exc
        self._r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key):  # pragma: no cover
        raw = self._r.get(key)
        return json.loads(raw) if raw else None

    def set(self, key, value, ttl_seconds):  # pragma: no cover
        self._r.set(key, json.dumps(value), ex=ttl_seconds)

    def delete(self, key):  # pragma: no cover
        self._r.delete(key)

    def clear(self):  # pragma: no cover
        self._r.flushdb()


def make_cache(backend: str, redis_url: str = "") -> CacheBackend:
    if backend == "memory":
        return InMemoryTTLCache()
    if backend == "null":
        return NullCache()
    if backend == "redis":
        return RedisCache(redis_url)
    raise ValueError(f"unknown cache backend '{backend}' (memory | null | redis)")

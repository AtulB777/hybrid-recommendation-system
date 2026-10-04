import pytest

from app.services.cache import InMemoryTTLCache, NullCache, make_cache


def test_set_get_roundtrip_and_miss():
    c = InMemoryTTLCache()
    c.set("a", {"x": [1, 2]}, 60)
    assert c.get("a") == {"x": [1, 2]}
    assert c.get("missing") is None
    assert (c.hits, c.misses) == (1, 1)


def test_ttl_expiry_uses_injected_clock():
    now = [0.0]
    c = InMemoryTTLCache(clock=lambda: now[0])
    c.set("k", {"v": 1}, ttl_seconds=10)
    now[0] = 9.9
    assert c.get("k") == {"v": 1}
    now[0] = 10.0
    assert c.get("k") is None


def test_delete_clear_and_eviction():
    c = InMemoryTTLCache(max_entries=2)
    c.set("a", {}, 60); c.set("b", {}, 60); c.set("c", {}, 60)
    assert c.get("a") is None and c.get("c") == {}
    c.delete("c"); assert c.get("c") is None
    c.set("z", {}, 60); c.clear(); assert c.get("z") is None


def test_values_must_be_json_serialisable():
    with pytest.raises(TypeError):
        InMemoryTTLCache().set("k", {"bad": object()}, 10)


def test_factory():
    assert isinstance(make_cache("memory"), InMemoryTTLCache)
    assert isinstance(make_cache("null"), NullCache)
    assert NullCache().get("x") is None
    with pytest.raises(ValueError):
        make_cache("memcached")

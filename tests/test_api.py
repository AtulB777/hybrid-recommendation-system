import re

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import create_app
from app.models.orm import Interaction
from app.services.cache import InMemoryTTLCache
from app.services.serving import RecommendationService
from tests.conftest import BATCH_K


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_get_recommendations_precomputed_then_cached(client, warm_user, model_bundle):
    r = client.get(f"/recommendations/{warm_user}?k=10")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "precomputed"
    assert body["model_version"] == model_bundle.version
    assert len(body["items"]) == 10
    assert [i["rank"] for i in body["items"]] == list(range(1, 11))
    item = body["items"][0]
    assert {"item_id", "title", "genres", "score", "explanation", "reason_type"} <= set(item)
    assert item["explanation"].startswith("Recommended because")
    assert client.get(f"/recommendations/{warm_user}?k=10").json()["source"] == "cache"


def test_precomputed_matches_realtime_scoring(client, warm_user):
    pre = client.get(f"/recommendations/{warm_user}?k=10").json()
    rt = client.post("/recommendations", json={"user_id": warm_user, "k": 10}).json()
    assert [i["item_id"] for i in pre["items"]] == [i["item_id"] for i in rt["items"]]
    assert rt["source"] == "realtime"


def test_k_beyond_precomputed_falls_back_to_realtime(client, warm_user):
    body = client.get(f"/recommendations/{warm_user}?k={BATCH_K + 5}").json()
    assert body["source"] == "realtime" and len(body["items"]) == BATCH_K + 5


def test_non_hybrid_model_and_exclude_seen_flag(client, warm_user):
    body = client.get(f"/recommendations/{warm_user}?k=5&model=content").json()
    assert body["source"] == "realtime" and body["model"] == "content"
    assert all(i["reason_type"] == "content" for i in body["items"])


def test_exclude_seen_is_honoured(client, warm_user):
    hist = {h["item_id"] for h in client.get(f"/users/{warm_user}/history?limit=500").json()["interactions"]}
    ids = {i["item_id"] for i in client.get(f"/recommendations/{warm_user}?k=20").json()["items"]}
    assert not hist & ids


def test_cold_user_gets_popularity(client, cold_user):
    body = client.get(f"/recommendations/{cold_user}?k=5").json()
    assert len(body["items"]) == 5
    assert all(i["reason_type"] == "popularity" for i in body["items"])


def test_error_handling(client):
    assert client.get("/recommendations/99999999").status_code == 404
    assert client.get("/recommendations/1?k=0").status_code == 422
    assert client.get("/recommendations/1?k=101").status_code == 422
    assert client.get("/recommendations/1?model=bogus").status_code == 400
    assert client.get("/items/99999999/similar").status_code == 404
    assert client.get("/users/99999999/history").status_code == 404
    assert client.get("/items/1/similar?method=bogus").status_code == 422


def test_post_recommendations_anonymous_session(client, model_bundle):
    f = model_bundle.feats
    sci_fi = [int(i) for i, g in zip(f.item_ids, f.items["genres_list"]) if g == ["sci-fi"] and not f.items.loc[i, "is_cold"]][:3]
    body = client.post("/recommendations", json={"history_item_ids": sci_fi, "k": 8}).json()
    ids = [i["item_id"] for i in body["items"]]
    assert len(ids) == 8 and not set(ids) & set(sci_fi)
    assert body["user_id"] is None
    assert any(i["reason_type"] != "popularity" for i in body["items"])
    for i in body["items"]:
        if i["reason_type"] != "popularity":
            assert re.match(r'^Recommended because you interacted with "', i["explanation"])


def test_post_empty_request_is_popularity_cold_start(client):
    body = client.post("/recommendations", json={"k": 5}).json()
    assert len(body["items"]) == 5 and all(i["reason_type"] == "popularity" for i in body["items"])


def test_post_exclude_items_and_validation(client, warm_user):
    first = client.post("/recommendations", json={"user_id": warm_user, "k": 5}).json()["items"]
    banned = [first[0]["item_id"]]
    again = client.post("/recommendations", json={"user_id": warm_user, "k": 5, "exclude_item_ids": banned}).json()["items"]
    assert banned[0] not in [i["item_id"] for i in again]
    assert client.post("/recommendations", json={"k": 0}).status_code == 422
    assert client.post("/recommendations", json={"user_id": 99999999}).status_code == 404
    assert client.post("/recommendations", json={"model": "nope"}).status_code == 400


def test_similar_items(client, model_bundle):
    f = model_bundle.feats
    warm_item = int(f.item_ids[~f.items["is_cold"].to_numpy()][0])
    cold_item = int(f.item_ids[f.items["is_cold"].to_numpy()][0])
    body = client.get(f"/items/{warm_item}/similar?k=5").json()
    assert len(body["items"]) == 5 and warm_item not in [i["item_id"] for i in body["items"]]
    assert body["is_cold_item"] is False
    cold = client.get(f"/items/{cold_item}/similar?k=5").json()
    assert cold["is_cold_item"] is True and len(cold["items"]) > 0
    assert client.get(f"/items/{cold_item}/similar?method=collaborative").json()["items"] == []


def test_user_history(client, warm_user, data):
    body = client.get(f"/users/{warm_user}/history?limit=5").json()
    expected_total = int((data.interactions["user_id"] == warm_user).sum())
    assert body["total_interactions"] == expected_total
    assert len(body["interactions"]) == min(5, expected_total)
    ts = [pd.Timestamp(i["timestamp"]) for i in body["interactions"]]
    assert ts == sorted(ts, reverse=True)
    assert len(body["top_genres"]) <= 3


def test_models_endpoint(client, model_bundle):
    body = client.get("/models").json()
    assert body["ready"] and body["model_version"] == model_bundle.version
    assert [m["name"] for m in body["models"]] == ["popularity", "content", "cf_item", "cf_user", "hybrid"]
    assert abs(sum(body["hybrid_weights"].values()) - 1) < 1e-9
    assert all(m["description"] for m in body["models"])


def test_evaluate_endpoint(client):
    payload = {"k_values": [5], "models": ["popularity", "hybrid"]}
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"]) == {"random", "popularity", "hybrid"}
    assert 0 <= body["results"]["hybrid"]["metrics"]["ndcg@5"] <= 1
    assert body["cached"] is False
    assert client.post("/evaluate", json=payload).json()["cached"] is True
    assert client.post("/evaluate", json={"k_values": [0]}).status_code == 422
    assert client.post("/evaluate", json={"models": ["nope"]}).status_code == 400


def test_fresh_interaction_bypasses_stale_precomputed(engine, model_bundle, warm_user):
    """A user with activity newer than the training data must be scored in real time."""
    uid = int(model_bundle.feats.users[model_bundle.feats.users["n_interactions"] >= 10].index[3])
    svc = RecommendationService(engine, InMemoryTTLCache(), model_bundle)
    with TestClient(create_app(service=svc)) as c:
        assert c.get(f"/recommendations/{uid}?k=5").json()["source"] == "precomputed"
        svc.cache.clear()
        unseen = next(int(i) for i in model_bundle.feats.item_ids
                      if int(i) not in {h["item_id"] for h in c.get(f"/users/{uid}/history?limit=500").json()["interactions"]})
        with svc.sessions() as s:
            s.add(Interaction(user_id=uid, item_id=unseen, event_type="purchase", weight=4.0,
                              timestamp=(model_bundle.data_as_of + pd.Timedelta(days=1)).to_pydatetime()))
            s.commit()
        try:
            body = c.get(f"/recommendations/{uid}?k=5").json()
            assert body["source"] == "realtime"
            assert unseen not in [i["item_id"] for i in body["items"]]
        finally:
            with svc.sessions() as s:
                s.execute(delete(Interaction).where(Interaction.user_id == uid, Interaction.item_id == unseen))
                s.commit()


def test_service_without_model_returns_503_but_history_still_works(engine, warm_user):
    svc = RecommendationService(engine, InMemoryTTLCache(), bundle=None)
    with TestClient(create_app(service=svc)) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get(f"/recommendations/{warm_user}").status_code == 503
        assert c.post("/recommendations", json={}).status_code == 503
        assert c.get("/models").json() == {**c.get("/models").json(), "ready": False, "models": []}
        assert c.get(f"/users/{warm_user}/history").status_code == 200

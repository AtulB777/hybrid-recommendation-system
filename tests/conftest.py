import pytest
from fastapi.testclient import TestClient

from app.database.session import init_db, make_engine
from app.main import create_app
from app.services.cache import InMemoryTTLCache
from app.services.ingestion import DataBundle, ingest_to_db, validate_bundle
from app.services.serving import RecommendationService
from app.services.synthetic_data import generate_synthetic_dataset
from app.services.training import fit_final_bundle, generate_batch_recommendations

BATCH_K = 20


@pytest.fixture(scope="session")
def data():
    raw = generate_synthetic_dataset(n_users=150, n_items=120, n_cold_users=5, n_cold_items=8, seed=7)
    clean, _ = validate_bundle(DataBundle(**raw))
    return clean


@pytest.fixture(scope="session")
def model_bundle(data):
    return fit_final_bundle(data)


@pytest.fixture(scope="session")
def engine(data, model_bundle):
    eng = make_engine("sqlite://")  # in-memory
    init_db(eng)
    ingest_to_db(data, eng)
    generate_batch_recommendations(model_bundle, eng, k=BATCH_K)
    return eng


@pytest.fixture
def service(engine, model_bundle):
    return RecommendationService(engine, InMemoryTTLCache(), model_bundle)


@pytest.fixture
def client(service):
    with TestClient(create_app(service=service)) as c:
        yield c


@pytest.fixture(scope="session")
def warm_user(model_bundle):
    feats = model_bundle.feats
    return int(feats.users[feats.users["n_interactions"] >= 10].index[0])


@pytest.fixture(scope="session")
def cold_user(model_bundle):
    feats = model_bundle.feats
    return int(feats.users[feats.users["n_interactions"] == 0].index[0])

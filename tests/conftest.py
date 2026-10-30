import pytest
from fastapi.testclient import TestClient

from src import train
from src.config import MODEL_PATH


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train.main()


@pytest.fixture(scope="session")
def client():
    from src.main import app

    # The context manager triggers the lifespan event, which loads the model.
    with TestClient(app) as test_client:
        yield test_client

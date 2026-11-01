import pytest
from src.model_training import load_experiment


@pytest.fixture(scope="session")
def bundle():
    return load_experiment()

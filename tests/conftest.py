import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "neighbour_impact_models.joblib"))
os.environ.setdefault("EXAMPLES_PATH", str(ROOT / "models" / "deploy_test_cases.json"))
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
os.environ["API_KEY"] = "test-key"          # tests run with authentication switched on
os.environ["LOG_LEVEL"] = "WARNING"

DATA_DIR = Path(os.environ["DATA_DIR"])
HAS_DATA = (DATA_DIR / "meta.json").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"no data store in {DATA_DIR} (copy the notebook's app_data)")


@pytest.fixture(scope="session")
def cases():
    """25 test rows exported by the notebook (section 19b): raw inputs, expected features and predictions."""
    return json.loads(Path(os.environ["EXAMPLES_PATH"]).read_text())


@pytest.fixture(scope="session")
def lookup_cases():
    """Real test outages with the notebook's prediction for every neighbour (section 19c)."""
    return json.loads((DATA_DIR / "lookup_test_cases.json").read_text())


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth():
    return {"X-API-Key": "test-key"}

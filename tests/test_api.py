# tests/test_all.py

import pytest
from fastapi.testclient import TestClient
from services.api import app

client = TestClient(app)

# ---------------------------
# 1. Test API readiness
# ---------------------------
def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["ready"] is True
    assert "features_count" in json_data

# ---------------------------
# 2. Test data preprocessing (mock example)
# ---------------------------
# If you have a real preprocess function, import it:
# from data.clean_data import preprocess
def preprocess(input_data: dict) -> dict:
    """Mock preprocess function for testing"""
    # simulate adding a feature
    output = input_data.copy()
    output["feature1"] = float(output.get("feature1", 0))
    output["feature2"] = float(output.get("feature2", 0))
    return output

def test_preprocess():
    input_data = {"feature1": 1, "feature2": 2}
    output = preprocess(input_data)
    assert "feature1" in output
    assert "feature2" in output
    assert isinstance(output["feature1"], float)

# ---------------------------
# 3. Test model inference via API
# ---------------------------
def test_predict():
    # create mock input using app.state.feature_names
    feature_names = getattr(app.state, "feature_names", ["feature1", "feature2"])
    sample_input = {f: 1 for f in feature_names}

    response = client.post("/predict", json={"data": [sample_input]})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)
    assert len(json_data["predictions"]) == 1

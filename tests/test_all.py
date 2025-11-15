# tests/test_all.py

import pytest
from fastapi.testclient import TestClient
from services.api import app
import numpy as np
from unittest.mock import MagicMock

client = TestClient(app)

# ---------------------------
# Fixture: Mock model/scaler
# ---------------------------
@pytest.fixture(autouse=True)
def setup_mock_model():
    """
    Automatically runs before each test.
    Mocks the model, scaler, and app.state to make endpoints testable without real files.
    """
    # Mock model: return NumPy array to match .tolist() call
    app.state.model = MagicMock()
    app.state.model.predict.return_value = np.array([0.5])  # matches predict endpoint expectations

    # Mock scaler
    app.state.scaler = MagicMock()
    app.state.scaler.transform.side_effect = lambda X: np.array(X)

    # Mock feature names
    app.state.feature_names = ["feature1", "feature2"]

    # Set ready state to True
    app.state.ready = True

# ---------------------------
# 1. Test API readiness
# ---------------------------
def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["ready"] is True
    assert "features_count" in json_data
    assert json_data["features_count"] == 2

# ---------------------------
# 2. Test data preprocessing (mock example)
# ---------------------------
def preprocess(input_data: dict) -> dict:
    """
    Mock preprocess function for testing.
    Converts input values to floats.
    """
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
    assert isinstance(output["feature2"], float)

# ---------------------------
# 3. Test model inference via API
# ---------------------------
def test_predict():
    # Create sample input based on mocked feature names
    feature_names = getattr(app.state, "feature_names")
    sample_input = {f: 1 for f in feature_names}

    response = client.post("/predict", json={"data": [sample_input]})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)
    assert len(json_data["predictions"]) == 1
    assert json_data["predictions"][0] == 0.5  # matches mocked model

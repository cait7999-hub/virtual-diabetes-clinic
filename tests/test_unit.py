"""Unit tests for the API (run after `python train.py`)."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

PATIENT = {
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
    "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "model_version" in r.json()


def test_predict_returns_float():
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_missing_feature_returns_422():
    bad = {k: v for k, v in PATIENT.items() if k != "s6"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "s6" in str(r.json()["detail"])


def test_non_numeric_returns_422():
    r = client.post("/predict", json={**PATIENT, "bmi": "high"})
    assert r.status_code == 422


def test_extra_field_rejected():
    r = client.post("/predict", json={**PATIENT, "foo": 1})
    assert r.status_code == 422

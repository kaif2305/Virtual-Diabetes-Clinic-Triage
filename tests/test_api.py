# tests/test_api.py
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"

def test_predict():
    payload = {
      "age": 0.02,"sex": -0.044,"bmi": 0.06,"bp": -0.03,"s1": -0.02,"s2": 0.03,"s3": -0.02,"s4": 0.02,"s5": 0.02,"s6": -0.001
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
    assert isinstance(j["prediction"], float)

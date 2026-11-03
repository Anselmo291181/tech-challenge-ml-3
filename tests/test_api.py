import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_urgency_valid():
    payload = {"laudo_medico": "Paciente com quadro grave de dor no peito."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "urgencia" in data
    assert data["urgencia"] in ["normal", "atenção", "urgente"]

def test_predict_empty_payload():
    payload = {"laudo_medico": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as api


class FakeModel:
    # remplace le modèle MLflow, absent en CI
    def predict(self, x):
        return np.array([[0.1, 0.2, 0.7]])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: None)
    monkeypatch.setattr(api, "model", FakeModel())
    with TestClient(api.app) as test_client:
        yield test_client


def image_png():
    buffer = io.BytesIO()
    Image.new("RGB", (28, 28), "white").save(buffer, format="PNG")
    return {"file": ("test.png", buffer.getvalue(), "image/png")}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_renvoie_la_classe_la_plus_probable(client):
    response = client.post("/predict", files=image_png())
    assert response.status_code == 200
    assert response.json()["class_id"] == 2
    assert response.json()["confidence"] == pytest.approx(0.7)


def test_predict_sans_modele_repond_503(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    response = client.post("/predict", files=image_png())
    assert response.status_code == 503


def test_metrics_compte_les_predictions(client):
    client.post("/predict", files=image_png())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'http_requests_total{endpoint="/predict",status="200"}' in response.text

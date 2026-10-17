from fastapi.testclient import TestClient
from app.main import create_app
from app.domain.mp_neuron import MPNeuron

def test_health():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    app = create_app()
    app.state.model = MPNeuron(
        features=("worst perimeter", "worst radius", "worst concave points", "mean concave points"),
        cutoffs={
            "worst perimeter": 113.0,
            "worst radius": 16.8,
            "worst concave points": 0.144,
            "mean concave points": 0.05,
        },
        threshold=3,
    )
    client = TestClient(app)

    # Caso benigno (valores bajos)
    payload_benigno = {
        "worst_perimeter": 85.0,
        "worst_radius": 13.0,
        "worst_concave_points": 0.08,
        "mean_concave_points": 0.03,
    }
    res_b = client.post("/api/v1/predict", json=payload_benigno)
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert data_b["diagnosis"] == "benigno"
    assert data_b["score"] == 4
    assert data_b["threshold"] == 3
    assert "Modelo educativo" in data_b["disclaimer"]

    # Caso maligno (valores altos)
    payload_maligno = {
        "worst_perimeter": 150.0,
        "worst_radius": 22.0,
        "worst_concave_points": 0.25,
        "mean_concave_points": 0.12,
    }
    res_m = client.post("/api/v1/predict", json=payload_maligno)
    assert res_m.status_code == 200
    data_m = res_m.json()
    assert data_m["diagnosis"] == "maligno"
    assert data_m["score"] == 0
    assert data_m["threshold"] == 3

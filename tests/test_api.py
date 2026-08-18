# This is a pytest for the FASTAPI application, Its check the health check and prediction endpoint.
import pytest
from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

valid_penguin={
    "island": "Biscoe",
    "bill_length_mm": 46.1,
    "bill_depth_mm": 14.8,
    "flipper_length_mm": 211,
    "body_mass_g": 4800,
    "sex": "female",
}

def test_health_check() ->None:
    response=client.get("/health")
    assert response.status_code==200
    assert "status" in response.json()

def test_predict_returns_species_and_probabilities() -> None:
    response = client.post("/predict", json=valid_penguin)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_species"] == "Gentoo"
    assert set(result["probabilities"]) == {
        "Adelie",
        "Chinstrap",
        "Gentoo",
    }

    assert sum(result["probabilities"].values()) == pytest.approx(
        1.0,
        abs=0.001,
    )


def test_predict_rejects_negative_measurement() -> None:
    invalid_penguin = valid_penguin | {"body_mass_g": -100}

    response = client.post("/predict", json=invalid_penguin)

    assert response.status_code == 422


def test_predict_rejects_unknown_sex() -> None:
    invalid_penguin = valid_penguin | {"sex": "unknown"}

    response = client.post("/predict", json=invalid_penguin)

    assert response.status_code == 422

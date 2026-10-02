"""Integration tests for the FastAPI prediction endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class MockModel:
    def predict(self, df):
        return np.array([0.35] * len(df))

    def predict_proba(self, df):
        return np.column_stack([
            np.full(len(df), 0.65),
            np.full(len(df), 0.35),
        ])


class MockTransformer:
    def transform(self, df):
        from src.config import FEATURE_COLS
        out = pd.DataFrame(index=df.index)
        for col in FEATURE_COLS:
            out[col] = 0.0
        return out


def test_model_and_transformer_load_from_same_version(tmp_path):
    import src.serving.app as serving_app

    version = SimpleNamespace(version="7", source="runs:/run-7/model")
    client = MagicMock()
    client.get_latest_versions.return_value = [version]
    model = MockModel()
    transformer = MockTransformer()
    transformer_path = tmp_path / "feature_pipeline.pkl"
    load_model = MagicMock(return_value=model)
    download_artifacts = MagicMock(return_value=str(transformer_path))
    mlflow = SimpleNamespace(
        set_tracking_uri=MagicMock(),
        xgboost=SimpleNamespace(load_model=load_model),
        artifacts=SimpleNamespace(download_artifacts=download_artifacts),
    )

    with (
        patch.object(serving_app, "_state", {}),
        patch.object(serving_app, "mlflow", mlflow),
        patch.object(serving_app, "MlflowClient", return_value=client),
        patch.object(
            serving_app.FlightFeatureTransformer,
            "load",
            return_value=transformer,
        ) as load_transformer,
    ):
        serving_app._load_model_and_transformer()

        load_model.assert_called_once_with("models:/flight-delay-model/7")
        download_artifacts.assert_called_once_with(
            artifact_uri="runs:/run-7/model/feature_pipeline.pkl",
            tracking_uri=serving_app.MLFLOW_TRACKING_URI,
        )
        load_transformer.assert_called_once_with(transformer_path)
        assert serving_app._state == {
            "model": model,
            "transformer": transformer,
            "model_version": "7",
        }


@pytest.fixture
def client():
    import src.serving.app as serving_app

    with patch.object(
        serving_app,
        "_state",
        {
            "model": MockModel(),
            "transformer": MockTransformer(),
            "model_version": "5",
            "start_time": 0.0,
            "prediction_logger": None,
        },
    ):
        with TestClient(serving_app.app) as c:
            yield c


VALID_PAYLOAD = {
    "month": 6,
    "day_of_week": 3,
    "dep_hour": 9,
    "carrier": "AA",
    "origin": "JFK",
    "dest": "LAX",
    "distance": 2475.0,
    "crs_elapsed_time": 330.0,
}


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data


class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "prediction" in data
        pred = data["prediction"]
        assert "delay_probability" in pred
        assert "is_delayed" in pred
        assert "confidence" in pred

    def test_probability_in_range(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["prediction"]["delay_probability"]
        assert 0.0 <= prob <= 1.0

    def test_invalid_month_rejected(self, client):
        payload = {**VALID_PAYLOAD, "month": 13}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_dep_hour_rejected(self, client):
        payload = {**VALID_PAYLOAD, "dep_hour": 25}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_carrier_uppercased(self, client):
        payload = {**VALID_PAYLOAD, "carrier": "aa"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["input"]["carrier"] == "AA"

    def test_confidence_tier_valid(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        confidence = resp.json()["prediction"]["confidence"]
        assert confidence in ("high", "medium", "low")

    def test_logs_engineered_features(self, client):
        import src.serving.app as serving_app
        from src.config import FEATURE_COLS

        prediction_logger = MagicMock()
        serving_app._state["prediction_logger"] = prediction_logger

        resp = client.post("/predict", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        logged_features = prediction_logger.log.call_args.kwargs["features"]
        assert set(logged_features) == set(FEATURE_COLS)


class TestBatchPredict:
    def test_batch_predict_returns_200(self, client):
        payload = {"flights": [VALID_PAYLOAD, VALID_PAYLOAD]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_response_count_matches_input(self, client):
        payload = {"flights": [VALID_PAYLOAD] * 5}
        resp = client.post("/predict/batch", json=payload)
        data = resp.json()
        assert data["total"] == 5
        assert len(data["predictions"]) == 5

    def test_empty_batch_rejected(self, client):
        payload = {"flights": []}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 422

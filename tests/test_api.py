"""Integration tests for the FastAPI prediction endpoint."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class MockModel:
    def predict(self, df):
        return np.array([0.35] * len(df))

    class _model_impl:
        @staticmethod
        def predict_proba(df):
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


@pytest.fixture
def client():
    with (
        patch("src.serving.app._state", {
            "model": MockModel(),
            "transformer": MockTransformer(),
            "model_version": "5",
            "start_time": 0.0,
            "prediction_logger": None,
        }),
    ):
        from src.serving.app import app
        with TestClient(app) as c:
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

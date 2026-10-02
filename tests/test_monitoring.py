"""Tests for monitoring data and report integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.config import FEATURE_COLS
from src.monitoring import drift
from src.monitoring import logger as monitoring_logger


def _feature_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({col: range(rows) for col in FEATURE_COLS})


def test_reference_snapshot_uses_reference_directory(tmp_path):
    reference_path = tmp_path / "reference" / "reference_snapshot.parquet"
    features = _feature_frame()

    with patch.object(drift, "REFERENCE_PARQUET", reference_path):
        saved_path = drift.save_reference_snapshot(features)

    assert saved_path == reference_path
    assert reference_path.exists()
    saved = pd.read_parquet(reference_path)
    assert list(saved.columns) == FEATURE_COLS


def test_fallback_report_matches_dashboard_schema(tmp_path):
    reference = _feature_frame(100)
    current = _feature_frame(100) + 100

    with patch.object(drift, "REPORTS_DIR", tmp_path):
        summary = drift._fallback_drift_report(
            reference,
            current,
            FEATURE_COLS,
            psi_threshold=0.2,
            pred_threshold=0.1,
            report_name="weekly_drift",
        )

    assert summary["dataset_drift_detected"] is True
    assert summary["drifted_columns_count"] == len(FEATURE_COLS)
    assert summary["total_columns"] == len(FEATURE_COLS)
    assert set(summary["column_drift"]) == set(FEATURE_COLS)
    assert summary["prediction_drift_score"] is None
    assert summary["retrain_recommended"] is True


def test_generated_report_writes_dashboard_summary(tmp_path):
    reference = _feature_frame(100)
    current = _feature_frame(100)

    with patch.object(drift, "REPORTS_DIR", tmp_path):
        summary = drift.generate_drift_report(
            reference,
            current,
            report_name="weekly_drift",
        )

    summary_path = tmp_path / "weekly_drift_summary.json"
    assert summary_path.exists()
    assert {
        "timestamp",
        "dataset_drift_detected",
        "drifted_columns_count",
        "total_columns",
        "column_drift",
        "prediction_drift_score",
        "retrain_recommended",
    }.issubset(summary)


def test_prediction_logger_writes_engineered_features(tmp_path):
    predictions_path = tmp_path / "predictions_log.parquet"
    features = {col: index for index, col in enumerate(FEATURE_COLS)}
    raw_input = {"carrier": "AA", "origin": "JFK", "dest": "LAX"}

    with patch.object(monitoring_logger, "PREDICTIONS_PARQUET", predictions_path):
        monitoring_logger.PredictionLogger().log(
            features=features,
            prediction=0.75,
            model_version="7",
            raw_input=raw_input,
        )

    saved = pd.read_parquet(predictions_path)
    assert set(FEATURE_COLS).issubset(saved.columns)
    assert saved.loc[0, "route_encoded"] == features["route_encoded"]
    assert saved.loc[0, "carrier"] == "AA"


def test_ingest_script_creates_reference_snapshot():
    import scripts.ingest as ingest

    feature_df = _feature_frame()
    feature_df["is_delayed"] = 0
    transformer = MagicMock()

    with (
        patch.object(ingest, "download_range", return_value=["raw.parquet"]),
        patch.object(ingest, "load_raw", return_value=pd.DataFrame()),
        patch.object(
            ingest,
            "validate",
            return_value=SimpleNamespace(row_count_raw=4, warnings=[]),
        ),
        patch.object(ingest, "clean", return_value=pd.DataFrame()),
        patch.object(ingest, "build_features", return_value=(feature_df, transformer)),
        patch.object(ingest, "save_processed"),
        patch.object(ingest, "save_reference_snapshot") as save_reference_snapshot,
    ):
        ingest.main(n_months=1)

    save_reference_snapshot.assert_called_once_with(feature_df)

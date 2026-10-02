"""Prediction logger — appends each API prediction to a local Parquet file."""

from __future__ import annotations

import fcntl
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import psycopg2

from src.config import DATABASE_URL, DATA_PROCESSED_DIR, FEATURE_COLS
from src.logger import get_logger

log = get_logger(__name__)

PREDICTIONS_PARQUET = DATA_PROCESSED_DIR / "predictions_log.parquet"
_METADATA_COLUMNS = [
    "id",
    "predicted_at",
    "model_version",
    "delay_probability",
    "is_delayed",
]
_RAW_COLUMNS = ["carrier", "origin", "dest"]
_PREDICTION_COLUMNS = _METADATA_COLUMNS + FEATURE_COLS + _RAW_COLUMNS

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id UUID PRIMARY KEY,
    predicted_at TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    delay_probability DOUBLE PRECISION NOT NULL,
    is_delayed BOOLEAN NOT NULL,
    month DOUBLE PRECISION NOT NULL,
    day_of_week DOUBLE PRECISION NOT NULL,
    dep_hour DOUBLE PRECISION NOT NULL,
    is_weekend DOUBLE PRECISION NOT NULL,
    season DOUBLE PRECISION NOT NULL,
    carrier_encoded DOUBLE PRECISION NOT NULL,
    origin_encoded DOUBLE PRECISION NOT NULL,
    dest_encoded DOUBLE PRECISION NOT NULL,
    route_encoded DOUBLE PRECISION NOT NULL,
    distance DOUBLE PRECISION NOT NULL,
    crs_elapsed_time DOUBLE PRECISION NOT NULL,
    dep_time_bucket DOUBLE PRECISION NOT NULL,
    carrier TEXT,
    origin TEXT,
    dest TEXT
)
"""
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS prediction_logs_predicted_at_idx
ON prediction_logs (predicted_at DESC)
"""
_INSERT_SQL = f"""
INSERT INTO prediction_logs ({", ".join(_PREDICTION_COLUMNS)})
VALUES ({", ".join(["%s"] * len(_PREDICTION_COLUMNS))})
"""
_SELECT_SQL = f"""
SELECT {", ".join(_PREDICTION_COLUMNS)}
FROM prediction_logs
ORDER BY predicted_at DESC
LIMIT %s
"""


class PredictionLogger:
    """Logs predictions to PostgreSQL or a locked local Parquet fallback."""

    def __init__(self, database_url: Optional[str] = DATABASE_URL) -> None:
        self.database_url = database_url

    def log(
        self,
        features: dict[str, Any],
        prediction: float,
        model_version: Optional[str] = None,
        raw_input: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "id": str(uuid.uuid4()),
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version or "unknown",
            "delay_probability": round(prediction, 4),
            "is_delayed": float(prediction >= 0.5),
        }
        record.update({col: float(features.get(col, 0)) for col in FEATURE_COLS})
        record.update({col: raw_input.get(col, "") if raw_input else "" for col in _RAW_COLUMNS})

        if self.database_url:
            try:
                self._write_database(record)
                return
            except Exception as exc:
                log.warning(
                    "Database prediction logging unavailable; using Parquet fallback",
                    error=str(exc),
                )

        try:
            self._write_parquet(record)
        except Exception as exc:
            log.error("Failed to write prediction log", error=str(exc))

    def load_predictions(self, limit: int = 5000) -> pd.DataFrame:
        """Load recent predictions for drift analysis."""
        if self.database_url:
            try:
                return self._load_database(limit)
            except Exception as exc:
                log.warning(
                    "Database prediction log unavailable; using Parquet fallback",
                    error=str(exc),
                )

        try:
            return self._load_parquet(limit)
        except Exception as exc:
            log.error("Failed to load prediction log", error=str(exc))
            return pd.DataFrame(columns=_PREDICTION_COLUMNS)

    def _ensure_database(self, connection) -> None:
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_TABLE_SQL)
            cursor.execute(_CREATE_INDEX_SQL)

    def _write_database(self, record: dict[str, Any]) -> None:
        values = [record[col] for col in _PREDICTION_COLUMNS]
        values[_PREDICTION_COLUMNS.index("is_delayed")] = bool(record["is_delayed"])
        with psycopg2.connect(self.database_url) as connection:
            self._ensure_database(connection)
            with connection.cursor() as cursor:
                cursor.execute(_INSERT_SQL, values)

    def _load_database(self, limit: int) -> pd.DataFrame:
        with psycopg2.connect(self.database_url) as connection:
            self._ensure_database(connection)
            with connection.cursor() as cursor:
                cursor.execute(_SELECT_SQL, (limit,))
                rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=_PREDICTION_COLUMNS)

    def _write_parquet(self, record: dict[str, Any]) -> None:
        PREDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        lock_path = PREDICTIONS_PARQUET.with_suffix(".lock")
        temp_path = PREDICTIONS_PARQUET.with_name(
            f".{PREDICTIONS_PARQUET.name}.{uuid.uuid4().hex}.tmp"
        )

        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                new_row = pd.DataFrame([record])
                if PREDICTIONS_PARQUET.exists():
                    existing = pd.read_parquet(PREDICTIONS_PARQUET)
                    combined = pd.concat([existing, new_row], ignore_index=True)
                else:
                    combined = new_row
                combined.to_parquet(temp_path, index=False, compression="snappy")
                os.replace(temp_path, PREDICTIONS_PARQUET)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_parquet(self, limit: int) -> pd.DataFrame:
        if not PREDICTIONS_PARQUET.exists():
            return pd.DataFrame(columns=_PREDICTION_COLUMNS)
        df = pd.read_parquet(PREDICTIONS_PARQUET)
        return df.sort_values("predicted_at", ascending=False).head(limit)

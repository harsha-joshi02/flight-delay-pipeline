"""Prediction logger — appends each API prediction to a local Parquet file."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.config import DATA_PROCESSED_DIR
from src.logger import get_logger

log = get_logger(__name__)

PREDICTIONS_PARQUET = DATA_PROCESSED_DIR / "predictions_log.parquet"


class PredictionLogger:
    """Logs predictions to a Parquet file for drift monitoring."""

    def log(
        self,
        features: dict[str, Any],
        prediction: float,
        model_version: Optional[str] = None,
    ) -> None:
        record = {
            "id": str(uuid.uuid4()),
            "predicted_at": datetime.utcnow().isoformat(),
            "model_version": model_version or "unknown",
            "carrier": features.get("carrier", ""),
            "origin": features.get("origin", ""),
            "dest": features.get("dest", ""),
            "month": float(features.get("month", 0)),
            "day_of_week": float(features.get("day_of_week", 0)),
            "dep_hour": float(features.get("dep_hour", 0)),
            "distance": float(features.get("distance", 0)),
            "crs_elapsed_time": float(features.get("crs_elapsed_time", 0)),
            "delay_probability": round(prediction, 4),
            "is_delayed": float(prediction >= 0.5),
        }
        try:
            new_row = pd.DataFrame([record])
            if PREDICTIONS_PARQUET.exists():
                existing = pd.read_parquet(PREDICTIONS_PARQUET)
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row
            combined.to_parquet(PREDICTIONS_PARQUET, index=False, compression="snappy")
        except Exception as exc:
            log.error("Failed to write prediction log", error=str(exc))

    def load_predictions(self, limit: int = 5000) -> pd.DataFrame:
        """Load recent predictions for drift analysis."""
        if not PREDICTIONS_PARQUET.exists():
            return pd.DataFrame()
        df = pd.read_parquet(PREDICTIONS_PARQUET)
        return df.sort_values("predicted_at", ascending=False).head(limit)

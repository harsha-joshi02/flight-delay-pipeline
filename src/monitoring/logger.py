"""Prediction logger — appends each API prediction to a local Parquet file."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.config import DATA_PROCESSED_DIR, FEATURE_COLS
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
        if raw_input:
            record.update({
                "carrier": raw_input.get("carrier", ""),
                "origin": raw_input.get("origin", ""),
                "dest": raw_input.get("dest", ""),
            })
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

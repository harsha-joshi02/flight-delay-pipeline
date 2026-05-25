"""Feature engineering: transforms raw BTS columns into model-ready features."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PROCESSED_DIR,
    DELAY_THRESHOLD_MINUTES,
    FEATURE_COLS,
    MODELS_DIR,
    TARGET_COL,
)
from src.logger import get_logger

log = get_logger(__name__)

PIPELINE_FILENAME = "feature_pipeline.pkl"


def _hhmm_to_hour(series: pd.Series) -> pd.Series:
    """Convert HHMM integer (e.g. 1435 → 14) to hour of day."""
    return (series.fillna(0).astype(int) // 100).clip(0, 23)


def _season(month: pd.Series) -> pd.Series:
    """Map month to season: 0=winter, 1=spring, 2=summer, 3=fall."""
    return pd.cut(
        month,
        bins=[0, 3, 6, 9, 12],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)


def _dep_time_bucket(hour: pd.Series) -> pd.Series:
    """Bucket departure hour: 0=night(0-5), 1=morning(6-11), 2=afternoon(12-17), 3=evening(18-23)."""
    return pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3],
    ).astype(int)


class FlightFeatureTransformer:
    """Stateful transformer fitted on training data and reused at serving time.

    Stores label encoders for categorical columns so training and inference
    use identical encodings.
    """

    def __init__(self) -> None:
        self._encoders: dict[str, LabelEncoder] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FlightFeatureTransformer":
        log.info("Fitting feature transformer", rows=len(df))
        df = self._base_transform(df)
        for col in ("carrier_encoded", "origin_encoded", "dest_encoded", "route_encoded"):
            enc = LabelEncoder()
            enc.fit(df[col].astype(str))
            self._encoders[col] = enc
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted. Call fit() first.")
        df = self._base_transform(df)
        df = self._encode_categoricals(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / PIPELINE_FILENAME)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        log.info("Feature transformer saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FlightFeatureTransformer":
        path = path or (MODELS_DIR / PIPELINE_FILENAME)
        if not path.exists():
            raise FileNotFoundError(f"Feature transformer not found at {path}")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        log.info("Feature transformer loaded", path=str(path))
        return obj

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        out["month"] = df["Month"].astype(int)
        out["day_of_week"] = df["DayOfWeek"].astype(int)
        out["is_weekend"] = (df["DayOfWeek"].isin([6, 7])).astype(int)
        out["season"] = _season(df["Month"])

        out["dep_hour"] = _hhmm_to_hour(df["CRSDepTime"])
        out["dep_time_bucket"] = _dep_time_bucket(out["dep_hour"])

        # Raw strings — label-encoded in _encode_categoricals
        out["carrier_encoded"] = df["Reporting_Airline"].astype(str).str.strip().str.upper()
        out["origin_encoded"] = df["Origin"].astype(str).str.strip().str.upper()
        out["dest_encoded"] = df["Dest"].astype(str).str.strip().str.upper()
        out["route_encoded"] = out["origin_encoded"] + "_" + out["dest_encoded"]

        out["distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
        out["crs_elapsed_time"] = pd.to_numeric(df["CRSElapsedTime"], errors="coerce").fillna(0)

        # Target only present in training data
        if "ArrDelay" in df.columns:
            out[TARGET_COL] = (df["ArrDelay"] >= DELAY_THRESHOLD_MINUTES).astype(int)

        return out

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, enc in self._encoders.items():
            if col not in df.columns:
                continue
            vals = df[col].astype(str)
            known = set(enc.classes_)
            vals = vals.where(vals.isin(known), other="__unknown__")
            if "__unknown__" not in known:
                enc.classes_ = np.append(enc.classes_, "__unknown__")
            df[col] = enc.transform(vals)
        return df


def build_features(
    df: pd.DataFrame,
    transformer: Optional[FlightFeatureTransformer] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, FlightFeatureTransformer]:
    if transformer is None:
        transformer = FlightFeatureTransformer()

    if fit:
        feature_df = transformer.fit_transform(df)
    else:
        feature_df = transformer.transform(df)

    log.info(
        "Feature engineering complete",
        rows=len(feature_df),
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        positive_rate=float(feature_df[TARGET_COL].mean()) if TARGET_COL in feature_df.columns else None,
    )
    return feature_df, transformer


def save_processed(df: pd.DataFrame, filename: str = "features.parquet") -> Path:
    path = DATA_PROCESSED_DIR / filename
    df.to_parquet(path, index=False, compression="snappy")
    log.info("Saved processed features", path=str(path), rows=len(df))
    return path


def load_processed(filename: str = "features.parquet") -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed features not found at {path}")
    return pd.read_parquet(path)

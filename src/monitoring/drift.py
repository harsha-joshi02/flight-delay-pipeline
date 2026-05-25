"""Drift monitoring: compares reference training data against recent predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_PROCESSED_DIR, FEATURE_COLS, REPORTS_DIR
from src.logger import get_logger

log = get_logger(__name__)

REFERENCE_PARQUET = DATA_PROCESSED_DIR / "reference_snapshot.parquet"
PREDICTIONS_PARQUET = DATA_PROCESSED_DIR / "predictions_log.parquet"

_DRIFT_PSI_THRESHOLD: float = 0.2
_PREDICTION_DRIFT_THRESHOLD: float = 0.1


def _load_env_thresholds() -> tuple[float, float]:
    import os
    psi = float(os.getenv("DRIFT_PSI_THRESHOLD", str(_DRIFT_PSI_THRESHOLD)))
    pred = float(os.getenv("PREDICTION_DRIFT_THRESHOLD", str(_PREDICTION_DRIFT_THRESHOLD)))
    return psi, pred


def save_reference_snapshot(df: pd.DataFrame) -> Path:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    snapshot = df[cols].copy()
    snapshot.to_parquet(REFERENCE_PARQUET, index=False, compression="snappy")
    log.info("Reference snapshot saved", path=str(REFERENCE_PARQUET), rows=len(snapshot))
    return REFERENCE_PARQUET


def load_reference() -> Optional[pd.DataFrame]:
    if not REFERENCE_PARQUET.exists():
        log.warning("No reference snapshot found", path=str(REFERENCE_PARQUET))
        return None
    return pd.read_parquet(REFERENCE_PARQUET)


def load_predictions(limit: int = 5000) -> Optional[pd.DataFrame]:
    if not PREDICTIONS_PARQUET.exists():
        log.warning("No prediction log found", path=str(PREDICTIONS_PARQUET))
        return None
    df = pd.read_parquet(PREDICTIONS_PARQUET)
    df = df.sort_values("predicted_at", ascending=False).head(limit)
    log.info("Loaded prediction log", rows=len(df))
    return df


def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    import numpy as np

    ref_vals = reference.dropna()
    cur_vals = current.dropna()
    if len(ref_vals) == 0 or len(cur_vals) == 0:
        return 0.0

    lo = min(ref_vals.min(), cur_vals.min())
    hi = max(ref_vals.max(), cur_vals.max())
    if lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    ref_counts = np.histogram(ref_vals, bins=edges)[0] + 1e-6
    cur_counts = np.histogram(cur_vals, bins=edges)[0] + 1e-6

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


def _wasserstein_distance(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import wasserstein_distance as wd
    return float(round(wd(a.dropna(), b.dropna()), 6))


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_name: str = "drift_report",
) -> dict:
    """Generate a drift report. Uses Evidently if available, falls back to PSI/Wasserstein."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    psi_threshold, pred_threshold = _load_env_thresholds()

    shared_cols = [c for c in FEATURE_COLS if c in reference.columns and c in current.columns]

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        ref_sub = reference[shared_cols].copy()
        cur_sub = current[shared_cols].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_sub, current_data=cur_sub)

        html_path = REPORTS_DIR / f"{report_name}.html"
        json_path = REPORTS_DIR / f"{report_name}.json"
        report.save_html(str(html_path))

        result_dict = report.as_dict()
        with open(json_path, "w") as fh:
            json.dump(result_dict, fh, indent=2, default=str)

        dataset_drift = False
        share_drifted = 0.0
        try:
            metrics = result_dict.get("metrics", [])
            for m in metrics:
                if "DatasetDriftMetric" in str(m.get("metric", "")):
                    dataset_drift = m["result"].get("dataset_drift", False)
                    share_drifted = m["result"].get("share_of_drifted_columns", 0.0)
                    break
        except Exception:
            pass

        pred_drift_score = 0.0
        if "delay_probability" in current.columns and "delay_probability" in reference.columns:
            pred_drift_score = _wasserstein_distance(
                reference["delay_probability"], current["delay_probability"]
            )

        summary = {
            "method": "evidently",
            "dataset_drift_detected": dataset_drift,
            "share_of_drifted_columns": share_drifted,
            "prediction_drift_score": pred_drift_score,
            "retrain_recommended": dataset_drift or (pred_drift_score > pred_threshold),
            "html_report": str(html_path),
            "json_report": str(json_path),
        }

    except ImportError:
        log.warning("Evidently not available — using fallback PSI/Wasserstein drift detection")
        summary = _fallback_drift_report(
            reference, current, shared_cols, psi_threshold, pred_threshold, report_name
        )

    log.info(
        "Drift report generated",
        dataset_drift=summary.get("dataset_drift_detected"),
        pred_drift=summary.get("prediction_drift_score"),
        retrain_recommended=summary.get("retrain_recommended"),
    )

    summary_path = REPORTS_DIR / f"{report_name}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


def _fallback_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cols: list[str],
    psi_threshold: float,
    pred_threshold: float,
    report_name: str,
) -> dict:
    psi_scores: dict[str, float] = {}
    for col in cols:
        if col in reference.columns and col in current.columns:
            psi_scores[col] = _compute_psi(reference[col], current[col])

    max_psi = max(psi_scores.values(), default=0.0)
    drifted_cols = [c for c, v in psi_scores.items() if v > psi_threshold]
    dataset_drift = len(drifted_cols) > len(cols) * 0.3

    pred_drift_score = 0.0
    if "delay_probability" in current.columns and "delay_probability" in reference.columns:
        pred_drift_score = _wasserstein_distance(
            reference["delay_probability"], current["delay_probability"]
        )

    json_path = REPORTS_DIR / f"{report_name}.json"
    result = {
        "method": "fallback_psi",
        "psi_scores": psi_scores,
        "max_psi": max_psi,
        "drifted_columns": drifted_cols,
        "dataset_drift_detected": dataset_drift,
        "share_of_drifted_columns": len(drifted_cols) / max(len(cols), 1),
        "prediction_drift_score": pred_drift_score,
        "retrain_recommended": dataset_drift or (pred_drift_score > pred_threshold),
        "json_report": str(json_path),
    }

    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)

    return result

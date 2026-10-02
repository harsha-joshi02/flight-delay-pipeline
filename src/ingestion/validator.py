"""Schema and data-quality validation for raw BTS flight data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)

_REQUIRED = {"Year", "Month", "Reporting_Airline", "Origin", "Dest", "ArrDelay", "Cancelled"}

_NON_NEGATIVE = {"Distance", "CRSElapsedTime"}

_BOUNDS: dict[str, tuple[Any, Any]] = {
    "Year": (1987, 2030),
    "Month": (1, 12),
    "DayofMonth": (1, 31),
    "DayOfWeek": (1, 7),
    "CRSDepTime": (0, 2359),
    "CRSArrTime": (0, 2359),
    "Distance": (0, 15_000),
}


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count_raw: int = 0
    row_count_after_filter: int = 0
    null_rates: dict[str, float] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate(df: pd.DataFrame, raise_on_error: bool = True) -> ValidationReport:
    report = ValidationReport(row_count_raw=len(df))

    missing_cols = _REQUIRED - set(df.columns)
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")

    if not report.passed:
        if raise_on_error:
            raise DataValidationError(report.errors)
        return report

    for col in df.columns:
        null_rate = df[col].isna().mean()
        report.null_rates[col] = round(float(null_rate), 4)
        if null_rate > 0.5:
            report.add_warning(f"Column '{col}' has {null_rate:.1%} null rate")

    if len(df) < 1000:
        report.add_warning(f"Very few rows ({len(df)}) — possible download issue")

    for col, (lo, hi) in _BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_bounds = df[col].dropna()
        out_of_bounds = out_of_bounds[(out_of_bounds < lo) | (out_of_bounds > hi)]
        if len(out_of_bounds) > 0:
            pct = len(out_of_bounds) / len(df)
            report.add_warning(
                f"Column '{col}' has {len(out_of_bounds)} rows ({pct:.1%}) outside [{lo}, {hi}]"
            )

    for col in _NON_NEGATIVE:
        if col not in df.columns:
            continue
        neg = (df[col] < 0).sum()
        if neg:
            report.add_warning(f"Column '{col}' has {neg} negative values")

    arr_null = df["ArrDelay"].isna().mean()
    if arr_null > 0.3:
        report.add_error(f"ArrDelay is {arr_null:.1%} null — too much missing target signal")

    id_cols = [c for c in ["Year", "Month", "DayofMonth", "Reporting_Airline",
                            "Flight_Number_Reporting_Airline", "Origin", "Dest"]
               if c in df.columns]
    if id_cols:
        n_dupes = df.duplicated(subset=id_cols).sum()
        if n_dupes > 0:
            report.add_warning(f"{n_dupes} duplicate rows detected")

    report.row_count_after_filter = len(df)

    for w in report.warnings:
        log.warning("Validation warning", msg=w)

    if report.errors:
        for e in report.errors:
            log.error("Validation error", msg=e)
        if raise_on_error:
            raise DataValidationError(report.errors)

    log.info(
        "Validation complete",
        passed=report.passed,
        rows=report.row_count_raw,
        warnings=len(report.warnings),
        errors=len(report.errors),
    )
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled/diverted flights and rows with missing or extreme ArrDelay."""
    original_len = len(df)

    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0].copy()
    if "Diverted" in df.columns:
        df = df[df["Diverted"] == 0].copy()

    df = df.dropna(subset=["ArrDelay"]).copy()
    df = df[df["ArrDelay"].between(-120, 1440)].copy()

    log.info(
        "Cleaning complete",
        original=original_len,
        remaining=len(df),
        dropped=original_len - len(df),
    )
    return df.reset_index(drop=True)

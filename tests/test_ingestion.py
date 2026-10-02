"""Unit tests for the ingestion validation and cleaning logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.validator import (
    DataValidationError,
    clean,
    validate,
)


def make_valid_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Year": 2023,
        "Month": rng.integers(1, 13, size=n),
        "DayofMonth": rng.integers(1, 29, size=n),
        "DayOfWeek": rng.integers(1, 8, size=n),
        "Reporting_Airline": rng.choice(["AA", "DL", "UA"], size=n),
        "Flight_Number_Reporting_Airline": rng.integers(100, 9999, size=n),
        "Origin": rng.choice(["JFK", "LAX", "ORD"], size=n),
        "Dest": rng.choice(["MIA", "DEN", "SEA"], size=n),
        "CRSDepTime": rng.integers(600, 2200, size=n),
        "ArrDelay": rng.normal(0, 20, size=n),
        "DepDelay": rng.normal(0, 20, size=n),
        "CRSElapsedTime": rng.integers(60, 360, size=n),
        "Distance": rng.integers(200, 3000, size=n),
        "Cancelled": 0,
        "Diverted": 0,
        "CarrierDelay": 0.0,
        "WeatherDelay": 0.0,
        "NASDelay": 0.0,
        "SecurityDelay": 0.0,
        "LateAircraftDelay": 0.0,
    })


class TestValidate:
    def test_valid_df_passes(self):
        df = make_valid_df()
        report = validate(df)
        assert report.passed

    def test_missing_required_column_raises(self):
        df = make_valid_df().drop(columns=["ArrDelay"])
        with pytest.raises(DataValidationError):
            validate(df, raise_on_error=True)

    def test_missing_required_column_no_raise(self):
        df = make_valid_df().drop(columns=["ArrDelay"])
        report = validate(df, raise_on_error=False)
        assert not report.passed
        assert any("ArrDelay" in e or "ArrDelay" in str(e) for e in report.errors + [str(report.errors)])

    @pytest.mark.parametrize(
        "column",
        ["DayOfWeek", "CRSDepTime", "Distance", "CRSElapsedTime"],
    )
    def test_feature_input_columns_are_required(self, column):
        df = make_valid_df().drop(columns=[column])

        with pytest.raises(DataValidationError, match=column):
            validate(df, raise_on_error=True)

    def test_high_null_rate_generates_warning(self):
        df = make_valid_df(500)
        df.loc[:400, "Distance"] = np.nan
        report = validate(df)
        assert any("Distance" in w for w in report.warnings)

    def test_out_of_bounds_month_generates_warning(self):
        df = make_valid_df(200)
        df.loc[0, "Month"] = 99
        report = validate(df)
        assert any("Month" in w for w in report.warnings)

    def test_very_few_rows_generates_warning(self):
        df = make_valid_df(50)
        report = validate(df)
        assert any("rows" in w.lower() or "few" in w.lower() for w in report.warnings)

    def test_high_arr_delay_null_rate_raises(self):
        df = make_valid_df(200)
        df.loc[:160, "ArrDelay"] = np.nan
        with pytest.raises(DataValidationError):
            validate(df, raise_on_error=True)

    def test_report_has_correct_row_count(self):
        df = make_valid_df(300)
        report = validate(df)
        assert report.row_count_raw == 300

    def test_null_rates_recorded(self):
        df = make_valid_df(200)
        df.loc[:99, "Distance"] = np.nan
        report = validate(df, raise_on_error=False)
        assert "Distance" in report.null_rates
        assert report.null_rates["Distance"] == pytest.approx(0.5, abs=0.01)


class TestClean:
    def test_cancelled_flights_removed(self):
        df = make_valid_df(200)
        df.loc[:99, "Cancelled"] = 1
        cleaned = clean(df)
        assert (cleaned["Cancelled"] == 0).all()

    def test_diverted_flights_removed(self):
        df = make_valid_df(200)
        df.loc[:49, "Diverted"] = 1
        cleaned = clean(df)
        assert (cleaned["Diverted"] == 0).all()

    def test_missing_arr_delay_removed(self):
        df = make_valid_df(200)
        df.loc[:29, "ArrDelay"] = np.nan
        cleaned = clean(df)
        assert cleaned["ArrDelay"].isna().sum() == 0

    def test_extreme_arr_delay_removed(self):
        df = make_valid_df(200)
        df.loc[0, "ArrDelay"] = 9999
        df.loc[1, "ArrDelay"] = -500
        cleaned = clean(df)
        assert (cleaned["ArrDelay"] <= 1440).all()
        assert (cleaned["ArrDelay"] >= -120).all()

    def test_index_reset_after_clean(self):
        df = make_valid_df(200)
        df.loc[::2, "Cancelled"] = 1
        cleaned = clean(df)
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_clean_preserves_valid_rows(self):
        df = make_valid_df(200)
        cleaned = clean(df)
        assert len(cleaned) == len(df)

    def test_clean_returns_dataframe(self):
        df = make_valid_df(100)
        result = clean(df)
        assert isinstance(result, pd.DataFrame)

    def test_mixed_bad_data(self):
        df = make_valid_df(100)
        df.loc[:9, "Cancelled"] = 1
        df.loc[10:19, "Diverted"] = 1
        df.loc[20:29, "ArrDelay"] = np.nan
        cleaned = clean(df)
        assert 60 <= len(cleaned) <= 80

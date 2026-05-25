"""Unit tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FlightFeatureTransformer, build_features, _hhmm_to_hour, _season


def make_raw_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Year": 2023,
        "Month": rng.integers(1, 13, size=n),
        "DayofMonth": rng.integers(1, 29, size=n),
        "DayOfWeek": rng.integers(1, 8, size=n),
        "Reporting_Airline": rng.choice(["AA", "DL", "UA", "WN"], size=n),
        "Flight_Number_Reporting_Airline": rng.integers(100, 9999, size=n),
        "Origin": rng.choice(["JFK", "LAX", "ORD", "ATL"], size=n),
        "Dest": rng.choice(["SEA", "DEN", "MIA", "PHX"], size=n),
        "CRSDepTime": rng.integers(600, 2200, size=n),
        "ArrDelay": rng.normal(0, 30, size=n),
        "CRSElapsedTime": rng.integers(60, 360, size=n),
        "Distance": rng.integers(200, 3000, size=n),
        "Cancelled": 0,
        "Diverted": 0,
    })


class TestHelpers:
    def test_hhmm_to_hour_standard(self):
        s = pd.Series([600, 1200, 1435, 2359, 0])
        result = _hhmm_to_hour(s)
        assert list(result) == [6, 12, 14, 23, 0]

    def test_hhmm_to_hour_clips_to_23(self):
        s = pd.Series([2400, 9999])
        result = _hhmm_to_hour(s)
        assert all(result <= 23)
        assert all(result >= 0)

    def test_hhmm_handles_nan(self):
        s = pd.Series([600, None, 1200])
        result = _hhmm_to_hour(s)
        assert not result.isna().any()

    def test_season_mapping(self):
        months = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        seasons = _season(months)
        assert seasons[0] == 0   # Jan → winter
        assert seasons[5] == 1   # Jun → spring
        assert seasons[6] == 2   # Jul → summer
        assert seasons[11] == 3  # Dec → fall


class TestFlightFeatureTransformer:
    def test_fit_transform_returns_correct_columns(self):
        df = make_raw_df(200)
        transformer = FlightFeatureTransformer()
        result = transformer.fit_transform(df)

        from src.config import FEATURE_COLS, TARGET_COL
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"
        assert TARGET_COL in result.columns

    def test_no_nulls_in_output(self):
        df = make_raw_df(200)
        transformer = FlightFeatureTransformer()
        result = transformer.fit_transform(df)
        from src.config import FEATURE_COLS
        assert result[FEATURE_COLS].isna().sum().sum() == 0

    def test_target_is_binary(self):
        df = make_raw_df(200)
        transformer = FlightFeatureTransformer()
        result = transformer.fit_transform(df)
        assert set(result["is_delayed"].unique()).issubset({0, 1})

    def test_is_weekend_logic(self):
        df = make_raw_df(500)
        transformer = FlightFeatureTransformer()
        result = transformer.fit_transform(df)
        weekend_rows = result[df["DayOfWeek"].isin([6, 7])]
        assert (weekend_rows["is_weekend"] == 1).all()
        weekday_rows = result[~df["DayOfWeek"].isin([6, 7])]
        assert (weekday_rows["is_weekend"] == 0).all()

    def test_transform_without_fit_raises(self):
        df = make_raw_df(10)
        transformer = FlightFeatureTransformer()
        with pytest.raises(RuntimeError, match="fit"):
            transformer.transform(df)

    def test_unseen_category_handled_gracefully(self):
        train_df = make_raw_df(200)
        transformer = FlightFeatureTransformer()
        transformer.fit(train_df)

        test_df = make_raw_df(10)
        test_df["Reporting_Airline"] = "ZZ"
        result = transformer.transform(test_df)
        assert result["carrier_encoded"].notna().all()

    def test_save_and_load_roundtrip(self, tmp_path):
        df = make_raw_df(200)
        transformer = FlightFeatureTransformer()
        transformer.fit(df)
        save_path = tmp_path / "feature_pipeline.pkl"
        transformer.save(save_path)

        loaded = FlightFeatureTransformer.load(save_path)
        result_original = transformer.transform(make_raw_df(50))
        result_loaded = loaded.transform(make_raw_df(50))
        pd.testing.assert_frame_equal(result_original, result_loaded)

    def test_dep_time_bucket_range(self):
        df = make_raw_df(500)
        df["CRSDepTime"] = [300, 800, 1400, 2000] * 125
        transformer = FlightFeatureTransformer()
        result = transformer.fit_transform(df)
        assert result["dep_time_bucket"].isin([0, 1, 2, 3]).all()


class TestBuildFeatures:
    def test_returns_tuple_of_df_and_transformer(self):
        df = make_raw_df(200)
        feature_df, transformer = build_features(df)
        assert isinstance(feature_df, pd.DataFrame)
        assert isinstance(transformer, FlightFeatureTransformer)

    def test_reuses_existing_transformer(self):
        df = make_raw_df(200)
        _, first_transformer = build_features(df, fit=True)
        df2 = make_raw_df(50)
        result, _ = build_features(df2, transformer=first_transformer, fit=False)
        assert len(result) == 50

    def test_positive_rate_reasonable(self):
        df = make_raw_df(500)
        df.loc[:249, "ArrDelay"] = 30
        df.loc[250:, "ArrDelay"] = -5
        feature_df, _ = build_features(df)
        pos_rate = feature_df["is_delayed"].mean()
        assert 0.3 < pos_rate < 0.7

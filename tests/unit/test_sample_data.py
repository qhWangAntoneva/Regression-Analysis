"""Tests for sample data generators in src/utils/sample_data.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.sample_data import (
    get_sample_datasets,
    load_air_quality_data,
    load_housing_data,
    load_sample_dataset,
    load_wages_data,
)


class TestLoadHousingData:
    def test_returns_dataframe(self):
        df = load_housing_data()
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        df = load_housing_data()
        assert df.shape == (500, 7)

    def test_correct_columns(self):
        df = load_housing_data()
        expected_cols = [
            "price", "sqft", "bedrooms", "age",
            "location_score", "floor", "has_garage",
        ]
        assert list(df.columns) == expected_cols

    def test_has_missing_values(self):
        df = load_housing_data()
        assert df["age"].isna().sum() > 0

    def test_price_positive(self):
        df = load_housing_data()
        assert (df["price"] > 0).all()

    def test_coefficient_recoverable(self):
        """Loosely verify OLS recovers approximate DGP coefficients."""
        import statsmodels.api as sm

        df = load_housing_data().dropna()
        y = df["price"]
        X = sm.add_constant(
            df[["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"]]
        )
        model = sm.OLS(y, X).fit()
        coef = model.params

        # Known DGP: 50000 + 200*sqft + 15000*bedrooms - 1000*age
        #   + 30000*location + 10000*floor + 20000*garage + noise
        # Loose bounds (within 50% of true, generous for noise)
        assert 100 < coef["sqft"] < 300, f"sqft coef {coef['sqft']:.1f} out of range"
        assert 5000 < coef["bedrooms"] < 25000, f"bedrooms coef {coef['bedrooms']:.1f}"
        assert -2000 < coef["age"] < 0, f"age coef {coef['age']:.1f}"
        assert 15000 < coef["location_score"] < 45000, f"location coef {coef['location_score']:.1f}"
        assert 0 < coef["floor"] < 20000, f"floor coef {coef['floor']:.1f}"
        assert 0 < coef["has_garage"] < 40000, f"garage coef {coef['has_garage']:.1f}"


class TestLoadWagesData:
    def test_returns_dataframe(self):
        df = load_wages_data()
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        df = load_wages_data()
        assert df.shape == (400, 7)

    def test_has_missing_values(self):
        df = load_wages_data()
        assert df["years_at_company"].isna().sum() > 0

    def test_wage_positive(self):
        df = load_wages_data()
        assert (df["wage"] > 0).all()

    def test_education_categories(self):
        df = load_wages_data()
        expected = {"高中以下", "高中", "本科", "硕士", "博士"}
        assert set(df["education"].unique()) == expected

    def test_coefficient_recoverable(self):
        """Loosely verify OLS recovers approximate wage DGP coefficients."""
        import statsmodels.api as sm

        df = load_wages_data().dropna()
        # Encode education as numeric for recovery check
        edu_map = {"高中以下": 0, "高中": 1, "本科": 2, "硕士": 3, "博士": 4}
        df = df.copy()
        df["edu_years"] = df["education"].map(edu_map)

        y = df["wage"]
        X = sm.add_constant(df[["edu_years", "experience", "hours_per_week", "years_at_company"]])
        model = sm.OLS(y, X).fit()
        coef = model.params

        # edu premium ~5000 per level
        assert 2000 < coef["edu_years"] < 8000, f"edu coef {coef['edu_years']:.1f}"
        # experience ~500 per year
        assert 200 < coef["experience"] < 800, f"exp coef {coef['experience']:.1f}"


class TestLoadAirQualityData:
    def test_returns_dataframe(self):
        df = load_air_quality_data()
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        df = load_air_quality_data()
        assert df.shape == (300, 6)

    def test_correct_columns(self):
        df = load_air_quality_data()
        assert list(df.columns) == ["AQI", "PM2.5", "PM10", "temp", "humidity", "wind"]

    def test_aqi_in_range(self):
        df = load_air_quality_data()
        assert df["AQI"].between(0, 300).all()

    def test_has_missing_values(self):
        df = load_air_quality_data()
        missing_total = df["PM2.5"].isna().sum() + df["wind"].isna().sum()
        assert missing_total > 0


class TestLoadSampleDataset:
    def test_load_housing(self):
        df = load_sample_dataset("房价数据")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (500, 7)

    def test_load_wages(self):
        df = load_sample_dataset("工资数据")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (400, 7)

    def test_load_air_quality(self):
        df = load_sample_dataset("空气质量数据")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (300, 6)

    def test_nonexistent_dataset_raises(self):
        with pytest.raises(ValueError, match="未知数据集"):
            load_sample_dataset("nonexistent")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="未知数据集"):
            load_sample_dataset("")


class TestGetSampleDatasets:
    def test_returns_dict(self):
        info = get_sample_datasets()
        assert isinstance(info, dict)

    def test_expected_keys(self):
        info = get_sample_datasets()
        assert set(info.keys()) == {"房价数据", "工资数据", "空气质量数据"}

    def test_each_entry_has_required_fields(self):
        info = get_sample_datasets()
        for key, meta in info.items():
            assert "description" in meta, f"{key} missing description"
            assert "source" in meta, f"{key} missing source"
            assert "n_rows" in meta, f"{key} missing n_rows"
            assert "n_cols" in meta, f"{key} missing n_cols"

    def test_n_rows_match_actual(self):
        """Verify metadata n_rows matches actual loaded data."""
        mapping = {
            "房价数据": load_housing_data,
            "工资数据": load_wages_data,
            "空气质量数据": load_air_quality_data,
        }
        info = get_sample_datasets()
        for name, loader in mapping.items():
            assert len(loader()) == info[name]["n_rows"], f"{name} row count mismatch"

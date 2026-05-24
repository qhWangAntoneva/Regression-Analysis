# encoding: utf-8
"""Test the MissingValueHandler: analyze + handle for all strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.missing import MissingValueHandler


@pytest.fixture
def missing_df():
    """Create a DataFrame with known missing values for testing."""
    rng = np.random.default_rng(42)
    n = 100

    df = pd.DataFrame({
        "numeric1": rng.normal(0, 1, n),
        "numeric2": rng.uniform(0, 10, n),
        "categorical1": rng.choice(["A", "B", "C"], n),
        "binary1": rng.integers(0, 2, n),
    })

    # Inject missing values
    df.loc[0:4, "numeric1"] = np.nan  # 5 missing
    df.loc[5:14, "numeric2"] = np.nan  # 10 missing
    df.loc[15:16, "categorical1"] = np.nan  # 2 missing
    df.loc[17, "binary1"] = np.nan  # 1 missing

    return df


class TestMissingValueHandlerAnalyze:
    """Tests for MissingValueHandler.analyze()."""

    def test_analyze_returns_dict(self, missing_df):
        handler = MissingValueHandler()
        result = handler.analyze(missing_df)

        assert isinstance(result, dict)
        assert "total_rows" in result
        assert "total_columns" in result
        assert "total_missing" in result
        assert "columns" in result

    def test_analyze_total_rows(self, missing_df):
        handler = MissingValueHandler()
        result = handler.analyze(missing_df)
        assert result["total_rows"] == 100

    def test_analyze_total_columns(self, missing_df):
        handler = MissingValueHandler()
        result = handler.analyze(missing_df)
        assert result["total_columns"] == 4

    def test_analyze_total_missing(self, missing_df):
        handler = MissingValueHandler()
        result = handler.analyze(missing_df)
        # 5 + 10 + 2 + 1 = 18
        assert result["total_missing"] == 18

    def test_analyze_column_stats(self, missing_df):
        handler = MissingValueHandler()
        result = handler.analyze(missing_df)

        col_info = result["columns"]["numeric1"]
        assert col_info["count"] == 5
        assert col_info["percentage"] == 5.0
        assert col_info["dtype"] == "float64"
        assert col_info["warn"] is False  # 5% is not >5%
        assert col_info["critical"] is False  # 5% is not >20%

    def test_analyze_warn_threshold(self, missing_df):
        handler = MissingValueHandler()
        result = handler.analyze(missing_df)

        # numeric2 has 10 missing = 10%
        col_info = result["columns"]["numeric2"]
        assert col_info["warn"] is True  # >5%
        assert col_info["critical"] is False  # not >20%

    def test_analyze_critical_threshold(self, missing_df):
        handler = MissingValueHandler()
        # Create a column with >20% missing
        df = missing_df.copy()
        df.loc[0:24, "numeric1"] = np.nan  # 25 missing = 25%
        result = handler.analyze(df)
        col_info = result["columns"]["numeric1"]
        assert col_info["critical"] is True  # >20%

    def test_analyze_no_missing(self, missing_df):
        df_no_missing = missing_df.dropna()
        handler = MissingValueHandler()
        result = handler.analyze(df_no_missing)
        assert result["total_missing"] == 0
        for col_info in result["columns"].values():
            assert col_info["count"] == 0
            assert col_info["percentage"] == 0.0
            assert col_info["warn"] is False
            assert col_info["critical"] is False

    def test_analyze_empty_dataframe(self):
        df = pd.DataFrame()
        handler = MissingValueHandler()
        result = handler.analyze(df)
        assert result["total_rows"] == 0
        assert result["total_columns"] == 0
        assert result["total_missing"] == 0

    def test_analyze_all_missing(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
        handler = MissingValueHandler()
        result = handler.analyze(df)
        assert result["total_missing"] == 2
        assert result["columns"]["a"]["percentage"] == 100.0
        assert result["columns"]["a"]["critical"] is True


class TestMissingValueHandlerHandle:
    """Tests for MissingValueHandler.handle()."""

    def test_handle_drop_all(self, missing_df):
        handler = MissingValueHandler()
        result = handler.handle(missing_df, strategy="drop")
        # Dropping all rows with any missing
        assert len(result) < len(missing_df)

    def test_handle_drop_specific_columns(self, missing_df):
        handler = MissingValueHandler()
        # Only drop rows where numeric1 is missing
        result = handler.handle(missing_df, strategy="drop", columns=["numeric1"])
        assert len(result) == len(missing_df) - 5  # 5 missing in numeric1

    def test_handle_mean_fill(self, missing_df):
        handler = MissingValueHandler()
        result = handler.handle(missing_df, strategy="mean")

        # numeric1 missing should be filled with mean
        assert result["numeric1"].isna().sum() < missing_df["numeric1"].isna().sum()
        # numeric2 missing should be filled with mean
        assert result["numeric2"].isna().sum() < missing_df["numeric2"].isna().sum()

    def test_handle_mean_on_numeric(self, missing_df):
        handler = MissingValueHandler()
        original_mean = missing_df["numeric1"].mean()
        result = handler.handle(missing_df, strategy="mean")

        # Check the filled values approximately equal the mean
        filled_mask = missing_df["numeric1"].isna()
        filled_values = result.loc[filled_mask, "numeric1"]
        assert abs(filled_values.mean() - original_mean) < 0.01

    def test_handle_median_fill(self, missing_df):
        handler = MissingValueHandler()
        original_median = missing_df["numeric2"].median()
        result = handler.handle(missing_df, strategy="median")

        filled_mask = missing_df["numeric2"].isna()
        filled_values = result.loc[filled_mask, "numeric2"]
        assert abs(filled_values.mean() - original_median) < 0.01

    def test_handle_categorical_uses_mode(self, missing_df):
        handler = MissingValueHandler()
        # categorical1 has 2 missing, mode should be 'A', 'B', or 'C'
        result = handler.handle(missing_df, strategy="mean")

        # Missing values should be filled
        filled_mask = missing_df["categorical1"].isna()
        assert result.loc[filled_mask, "categorical1"].notna().all()
        # The filled values should be one of the categories
        for val in result.loc[filled_mask, "categorical1"]:
            assert val in ["A", "B", "C"]

    def test_handle_invalid_strategy(self, missing_df):
        handler = MissingValueHandler()
        with pytest.raises(ValueError, match="不支持的缺失值处理策略"):
            handler.handle(missing_df, strategy="invalid")

    def test_handle_no_missing_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        handler = MissingValueHandler()
        result = handler.handle(df, strategy="mean")
        pd.testing.assert_frame_equal(df, result)

    def test_handle_mean_on_binary_col(self, missing_df):
        handler = MissingValueHandler()
        # binary1 has 1 missing
        result = handler.handle(missing_df, strategy="mean")
        assert result["binary1"].isna().sum() < missing_df["binary1"].isna().sum()

    def test_handle_drop_no_missing_in_col(self, missing_df):
        handler = MissingValueHandler()
        # Binary column with no missing
        no_missing_col = "no_missing"
        missing_df[no_missing_col] = 1.0
        result = handler.handle(missing_df, strategy="drop", columns=[no_missing_col])
        assert len(result) == len(missing_df)  # No rows dropped

    def test_handle_mean_preserves_dtype(self, missing_df):
        handler = MissingValueHandler()
        result = handler.handle(missing_df, strategy="mean")
        assert result["numeric1"].dtype == missing_df["numeric1"].dtype
        assert result["binary1"].dtype == missing_df["binary1"].dtype

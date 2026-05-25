# encoding: utf-8
"""Unit tests for src.results.statistics module.

Covers descriptive_stats() edge cases and correlation_matrix() completely.
Existing basic tests for descriptive_stats(), anova_oneway(), and freq_table()
are in test_fitter.py and test_results_phase2.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.results.statistics import (
    anova_oneway,
    correlation_matrix,
    descriptive_stats,
    freq_table,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_ols.csv"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load the sample OLS test dataset."""
    return pd.read_csv(SAMPLE_CSV, encoding="utf-8")


# =========================================================================
# descriptive_stats edge cases
# =========================================================================

class TestDescriptiveStatsEdgeCases:
    """Edge cases for descriptive_stats()."""

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame: all columns should have 0 observations."""
        data = pd.DataFrame({"x": pd.Series([], dtype=float)})
        stats = descriptive_stats(data, ["x"])
        assert len(stats) == 1
        assert stats.loc["x", "观测数"] == 0
        assert stats.loc["x", "缺失率"] == 1.0
        assert pd.isna(stats.loc["x", "均值"])

    def test_empty_dataframe_no_rows(self) -> None:
        """DataFrame with no rows but column defined."""
        data = pd.DataFrame(columns=["a", "b"])
        stats = descriptive_stats(data, ["a", "b"])
        assert len(stats) == 2
        for col in ["a", "b"]:
            assert stats.loc[col, "观测数"] == 0
            assert stats.loc[col, "缺失率"] == 1.0

    def test_single_column(self) -> None:
        """Single numeric column."""
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        stats = descriptive_stats(data, ["x"])
        assert len(stats) == 1
        assert stats.loc["x", "观测数"] == 5
        assert stats.loc["x", "均值"] == 3.0
        assert stats.loc["x", "最小值"] == 1.0
        assert stats.loc["x", "最大值"] == 5.0
        assert stats.loc["x", "缺失值数"] == 0
        assert stats.loc["x", "缺失率"] == 0.0

    def test_all_nan_column(self) -> None:
        """Column where every value is NaN."""
        data = pd.DataFrame({
            "all_nan": [np.nan, np.nan, np.nan],
            "normal": [1.0, 2.0, 3.0],
        })
        stats = descriptive_stats(data, ["all_nan", "normal"])
        # all_nan column
        assert stats.loc["all_nan", "观测数"] == 0
        assert stats.loc["all_nan", "缺失值数"] == 3
        assert stats.loc["all_nan", "缺失率"] == 1.0
        assert pd.isna(stats.loc["all_nan", "均值"])
        # normal column
        assert stats.loc["normal", "观测数"] == 3
        assert stats.loc["normal", "缺失值数"] == 0

    def test_partial_nan_column(self) -> None:
        """Column with some NaN values."""
        data = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        stats = descriptive_stats(data, ["x"])
        assert stats.loc["x", "观测数"] == 3  # 3 non-missing
        assert stats.loc["x", "缺失值数"] == 2
        assert stats.loc["x", "缺失率"] == 0.4
        assert stats.loc["x", "均值"] == 3.0

    def test_categorical_column(self) -> None:
        """Categorical (string) column should fall back to NaN stats."""
        data = pd.DataFrame({"cat": ["A", "B", "A", "C", "B"]})
        stats = descriptive_stats(data, ["cat"])
        assert stats.loc["cat", "观测数"] == 5
        assert stats.loc["cat", "缺失值数"] == 0
        # Numeric stats should be NaN for categorical
        assert pd.isna(stats.loc["cat", "均值"])
        assert pd.isna(stats.loc["cat", "标准差"])

    def test_mixed_types(self) -> None:
        """Mix of numeric and categorical variables."""
        data = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["A", "B", "C"],
        })
        stats = descriptive_stats(data, ["num", "cat"])
        assert len(stats) == 2
        # Numeric
        assert not pd.isna(stats.loc["num", "均值"])
        # Categorical
        assert pd.isna(stats.loc["cat", "均值"])

    def test_single_row(self) -> None:
        """Single row of data should still compute."""
        data = pd.DataFrame({"x": [5.0]})
        stats = descriptive_stats(data, ["x"])
        assert stats.loc["x", "观测数"] == 1
        assert stats.loc["x", "均值"] == 5.0
        assert stats.loc["x", "最小值"] == 5.0
        assert stats.loc["x", "最大值"] == 5.0
        assert stats.loc["x", "50%"] == 5.0
        # Standard deviation with ddof=1 on single value is NaN
        assert pd.isna(stats.loc["x", "标准差"])

    def test_negative_and_zero_values(self) -> None:
        """Variables with negative and zero values."""
        data = pd.DataFrame({"x": [-5.0, 0.0, 5.0]})
        stats = descriptive_stats(data, ["x"])
        assert stats.loc["x", "最小值"] == -5.0
        assert stats.loc["x", "最大值"] == 5.0
        assert stats.loc["x", "均值"] == 0.0

    def test_identical_values(self) -> None:
        """All identical values (zero variance)."""
        data = pd.DataFrame({"x": [7.0, 7.0, 7.0, 7.0]})
        stats = descriptive_stats(data, ["x"])
        assert stats.loc["x", "均值"] == 7.0
        assert stats.loc["x", "最小值"] == 7.0
        assert stats.loc["x", "最大值"] == 7.0
        assert stats.loc["x", "标准差"] == 0.0

    def test_large_values(self) -> None:
        """Large magnitude values."""
        data = pd.DataFrame({"x": [1e6, 2e6, 3e6]})
        stats = descriptive_stats(data, ["x"])
        assert stats.loc["x", "均值"] == 2e6
        assert stats.loc["x", "最小值"] == 1e6
        assert stats.loc["x", "最大值"] == 3e6

    def test_index_is_variable_name(self) -> None:
        """Result index should be the variable names."""
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        stats = descriptive_stats(data, ["a", "b"])
        assert list(stats.index) == ["a", "b"]


# =========================================================================
# correlation_matrix tests
# =========================================================================

class TestCorrelationMatrix:
    """Tests for correlation_matrix()."""

    def test_pearson_basic(self, sample_data: pd.DataFrame) -> None:
        """Basic Pearson correlation on numeric variables."""
        corr = correlation_matrix(sample_data, ["x1", "x2", "x4"])
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        # Diagonal should be 1.0
        for var in ["x1", "x2", "x4"]:
            assert abs(corr.loc[var, var] - 1.0) < 0.001
        # All values should be within [-1, 1]
        for var1 in ["x1", "x2", "x4"]:
            for var2 in ["x1", "x2", "x4"]:
                assert -1.0 <= corr.loc[var1, var2] <= 1.0

    def test_spearman_method(self, sample_data: pd.DataFrame) -> None:
        """Spearman rank correlation."""
        corr = correlation_matrix(sample_data, ["x1", "x2", "x4"], method="spearman")
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        for var in ["x1", "x2", "x4"]:
            assert abs(corr.loc[var, var] - 1.0) < 0.001

    def test_kendall_method(self, sample_data: pd.DataFrame) -> None:
        """Kendall tau correlation."""
        corr = correlation_matrix(sample_data, ["x1", "x2", "x4"], method="kendall")
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        for var in ["x1", "x2", "x4"]:
            assert abs(corr.loc[var, var] - 1.0) < 0.001

    def test_invalid_method_raises(self, sample_data: pd.DataFrame) -> None:
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            correlation_matrix(sample_data, ["x1", "x2"], method="invalid")

    def test_single_variable_raises(self, sample_data: pd.DataFrame) -> None:
        """Single variable should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 numeric"):
            correlation_matrix(sample_data, ["x1"])

    def test_missing_variables_raises(self, sample_data: pd.DataFrame) -> None:
        """Missing variables should raise ValueError."""
        with pytest.raises(ValueError, match="not found in data"):
            correlation_matrix(sample_data, ["x1", "nonexistent"])

    def test_constant_variable(self) -> None:
        """Constant variable (zero variance) should produce NaN correlation."""
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [7.0, 7.0, 7.0, 7.0, 7.0],
        })
        corr = correlation_matrix(data, ["a", "constant"])
        # Correlation with constant should be NaN
        assert pd.isna(corr.loc["a", "constant"])
        assert pd.isna(corr.loc["constant", "a"])
        # Diagonal should still be 1.0
        assert corr.loc["a", "a"] == 1.0

    def test_all_nan_column(self) -> None:
        """All-NaN column: correlation will be computed but results are NaN."""
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "all_nan": [np.nan] * 5,
        })
        corr = correlation_matrix(data, ["a", "all_nan"])
        # all_nan is numeric so it won't be excluded, but correlation will be NaN
        assert pd.isna(corr.loc["a", "all_nan"])
        assert pd.isna(corr.loc["all_nan", "a"])

    def test_with_non_numeric_variable(self, sample_data: pd.DataFrame) -> None:
        """Non-numeric variable should trigger warning and be excluded."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corr = correlation_matrix(sample_data, ["x1", "x2", "x3"])
            # x3 is categorical, should be excluded
            assert "x3" not in corr.columns
            assert "x1" in corr.columns
            assert "x2" in corr.columns
            # Warning should mention x3
            assert any("x3" in str(warning.message) for warning in w)

    def test_all_non_numeric_raises(self, sample_data: pd.DataFrame) -> None:
        """All non-numeric variables should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 numeric"):
            correlation_matrix(sample_data, ["x3"])  # categorical only

    def test_only_one_numeric_after_filter_raises(self, sample_data: pd.DataFrame) -> None:
        """Only one numeric after filtering out non-numeric should raise."""
        with pytest.raises(ValueError, match="at least 2 numeric"):
            correlation_matrix(sample_data, ["x1", "x3"])  # x1 numeric, x3 categorical

    def test_values_rounded_to_4_decimals(self, sample_data: pd.DataFrame) -> None:
        """Correlation values should be rounded to 4 decimal places."""
        corr = correlation_matrix(sample_data, ["x1", "x2", "x4"])
        for var1 in corr.columns:
            for var2 in corr.columns:
                val = corr.loc[var1, var2]
                if not pd.isna(val):
                    assert val == round(val, 4)

    def test_symmetry(self, sample_data: pd.DataFrame) -> None:
        """Correlation matrix should be symmetric."""
        corr = correlation_matrix(sample_data, ["x1", "x2", "x4"])
        for var1 in corr.columns:
            for var2 in corr.columns:
                if var1 == var2:
                    continue
                val1 = corr.loc[var1, var2]
                val2 = corr.loc[var2, var1]
                if pd.isna(val1) and pd.isna(val2):
                    continue
                assert abs(val1 - val2) < 0.0001

    def test_two_variables_perfect_correlation(self) -> None:
        """Two perfectly correlated variables."""
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        corr = correlation_matrix(data, ["a", "b"])
        assert abs(corr.loc["a", "b"] - 1.0) < 0.001

    def test_two_variables_perfect_negative_correlation(self) -> None:
        """Two perfectly negatively correlated variables."""
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        corr = correlation_matrix(data, ["a", "b"])
        assert abs(corr.loc["a", "b"] - (-1.0)) < 0.001


# =========================================================================
# Additional anova_oneway edge cases
# =========================================================================

class TestAnovaOnewayEdgeCases:
    """Additional edge cases for anova_oneway()."""

    def test_anova_drop_nan_rows(self) -> None:
        """NaN rows should be dropped before ANOVA."""
        data = pd.DataFrame({
            "dv": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        })
        result = anova_oneway(data, dv="dv", group="group")
        assert result["group_counts"]["A"] == 2  # one NaN excluded
        assert result["group_counts"]["B"] == 3
        assert isinstance(result["f_statistic"], float)
        assert isinstance(result["p_value"], float)

    def test_anova_three_groups(self) -> None:
        """ANOVA with three groups."""
        data = pd.DataFrame({
            "dv": [1, 2, 1, 5, 6, 5, 9, 10, 11],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        })
        result = anova_oneway(data, dv="dv", group="group")
        assert len(result["group_means"]) == 3
        assert len(result["group_counts"]) == 3

    def test_anova_single_val_per_group_std_zero(self) -> None:
        """Groups with only one value should have std = 0."""
        data = pd.DataFrame({
            "dv": [1.0, 2.0, 3.0, 4.0],
            "group": ["A", "B", "C", "D"],
        })
        result = anova_oneway(data, dv="dv", group="group")
        assert result["df_within"] == 0  # n - k = 4 - 4 = 0
        for g in ["A", "B", "C", "D"]:
            assert result["group_stds"][g] == 0.0

    def test_anova_binary_group(self) -> None:
        """ANOVA with binary (0/1) group variable."""
        data = pd.DataFrame({
            "dv": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            "group": [0, 0, 0, 1, 1, 1],
        })
        result = anova_oneway(data, dv="dv", group="group")
        assert len(result["group_means"]) == 2
        assert "0" in result["group_means"] or 0 in result["group_means"]


# =========================================================================
# Additional freq_table edge cases
# =========================================================================

class TestFreqTableEdgeCases:
    """Additional edge cases for freq_table()."""

    def test_freq_table_single_category(self) -> None:
        """Only one unique value."""
        data = pd.DataFrame({"x": ["A", "A", "A"]})
        ft = freq_table(data, col="x")
        assert len(ft) == 1
        assert ft.loc[1, "频数"] == 3
        assert ft.loc[1, "百分比(%)"] == 100.0
        assert ft.loc[1, "累积百分比(%)"] == 100.0

    def test_freq_table_all_missing(self) -> None:
        """All values are NaN."""
        data = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        ft = freq_table(data, col="x")
        assert len(ft) == 0

    def test_freq_table_ordering(self) -> None:
        """Categories should be ordered by frequency (descending)."""
        data = pd.DataFrame({"x": ["C", "A", "A", "B", "A", "C"]})
        ft = freq_table(data, col="x")
        # A appears 3 times, C appears 2, B appears 1
        assert ft.loc[1, "类别"] == "A"
        assert ft.loc[1, "频数"] == 3
        assert ft.loc[2, "频数"] == 2
        assert ft.loc[3, "频数"] == 1

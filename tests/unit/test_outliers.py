# encoding: utf-8
"""Test the OutlierDetector: IQR, zscore, flag_outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.outliers import OutlierDetector


@pytest.fixture
def normal_df():
    """Create a DataFrame with normally distributed data and injected outliers."""
    rng = np.random.default_rng(42)
    n = 200

    # Normal data
    df = pd.DataFrame({
        "norm": rng.normal(0, 1, n),
        "uniform": rng.uniform(0, 10, n),
        "categorical": rng.choice(["A", "B", "C"], n),
    })

    # Inject extreme outliers
    df.loc[0, "norm"] = 100.0  # Extreme outlier
    df.loc[1, "norm"] = -100.0  # Extreme outlier
    df.loc[2, "norm"] = 15.0  # Strong outlier
    df.loc[0, "uniform"] = 1000.0  # Extreme outlier

    return df


@pytest.fixture
def clean_df():
    """A clean DataFrame with no outliers."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x": rng.normal(50, 5, n),
        "y": rng.uniform(20, 30, n),
    })


class TestOutlierDetectorIQR:
    """Tests for OutlierDetector.detect_iqr()."""

    def test_detect_iqr_returns_series(self, normal_df):
        detector = OutlierDetector()
        result = detector.detect_iqr(normal_df, "norm")
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_detect_iqr_finds_outliers(self, normal_df):
        detector = OutlierDetector()
        result = detector.detect_iqr(normal_df, "norm")
        # Should find at least the extreme value at index 0
        assert result.iloc[0] == True  # noqa: E712

    def test_detect_iqr_default_multiplier(self, normal_df):
        detector = OutlierDetector()
        result = detector.detect_iqr(normal_df, "norm")
        # Count outliers with default 1.5 multiplier
        n_outliers = result.sum()
        assert n_outliers >= 3  # At least our 3 injected outliers

    def test_detect_iqr_tighter_multiplier(self, normal_df):
        detector = OutlierDetector()
        # With multiplier=3.0, fewer values flagged
        strict = detector.detect_iqr(normal_df, "norm", multiplier=3.0)
        loose = detector.detect_iqr(normal_df, "norm", multiplier=1.5)
        assert strict.sum() <= loose.sum()

    def test_detect_iqr_raises_on_missing_column(self, normal_df):
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="不存在"):
            detector.detect_iqr(normal_df, "nonexistent")

    def test_detect_iqr_raises_on_non_numeric(self, normal_df):
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="不是数值类型"):
            detector.detect_iqr(normal_df, "categorical")

    def test_detect_iqr_no_outliers_in_clean(self, clean_df):
        detector = OutlierDetector()
        result = detector.detect_iqr(clean_df, "x")
        assert result.sum() == 0  # No outliers in clean data

    def test_detect_iqr_no_nans_in_result(self, normal_df):
        detector = OutlierDetector()
        result = detector.detect_iqr(normal_df, "norm")
        assert not result.isna().any()


class TestOutlierDetectorZScore:
    """Tests for OutlierDetector.detect_zscore()."""

    def test_detect_zscore_returns_series(self, normal_df):
        detector = OutlierDetector()
        result = detector.detect_zscore(normal_df, "norm")
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_detect_zscore_finds_extreme_outliers(self, normal_df):
        detector = OutlierDetector()
        result = detector.detect_zscore(normal_df, "norm", threshold=3.0)
        # Index 0 has value 100 -> z-score ~100 -> outlier
        assert result.iloc[0] == True  # noqa: E712

    def test_detect_zscore_lower_threshold_more_outliers(self, normal_df):
        detector = OutlierDetector()
        strict = detector.detect_zscore(normal_df, "norm", threshold=5.0)
        loose = detector.detect_zscore(normal_df, "norm", threshold=2.0)
        assert strict.sum() <= loose.sum()

    def test_detect_zscore_raises_on_missing_column(self, normal_df):
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="不存在"):
            detector.detect_zscore(normal_df, "nonexistent")

    def test_detect_zscore_raises_on_non_numeric(self, normal_df):
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="不是数值类型"):
            detector.detect_zscore(normal_df, "categorical")

    def test_detect_zscore_no_outliers_in_clean(self, clean_df):
        detector = OutlierDetector()
        result = detector.detect_zscore(clean_df, "y")
        assert result.sum() == 0

    def test_detect_zscore_zero_std_no_outliers(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
        detector = OutlierDetector()
        result = detector.detect_zscore(df, "x")
        assert result.sum() == 0
        assert len(result) == 4
        assert not result.any()


class TestOutlierDetectorFlagOutliers:
    """Tests for OutlierDetector.flag_outliers()."""

    def test_flag_outliers_returns_tuple(self, normal_df):
        detector = OutlierDetector()
        result = detector.flag_outliers(normal_df, ["norm", "uniform"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_flag_outliers_adds_outlier_columns(self, normal_df):
        detector = OutlierDetector()
        df_result, summary = detector.flag_outliers(normal_df, ["norm", "uniform"])
        assert "norm_outlier" in df_result.columns
        assert "uniform_outlier" in df_result.columns

    def test_flag_outliers_summary_keys(self, normal_df):
        detector = OutlierDetector()
        df_result, summary = detector.flag_outliers(normal_df, ["norm"])
        assert "norm" in summary
        assert "n_outliers" in summary["norm"]
        assert "percentage" in summary["norm"]
        assert summary["norm"]["n_outliers"] > 0

    def test_flag_outliers_original_data_preserved(self, normal_df):
        detector = OutlierDetector()
        df_result, summary = detector.flag_outliers(normal_df, ["norm"])
        for col in normal_df.columns:
            pd.testing.assert_series_equal(
                df_result[col], normal_df[col], check_names=False
            )

    def test_flag_outliers_with_zscore(self, normal_df):
        detector = OutlierDetector()
        df_result, summary = detector.flag_outliers(
            normal_df, ["norm"], method="zscore", threshold=3.0
        )
        assert "norm_outlier" in df_result.columns
        # The extreme outlier (100) should be detected
        assert df_result.loc[0, "norm_outlier"] == True  # noqa: E712

    def test_flag_outliers_invalid_method(self, normal_df):
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="不支持的检测方法"):
            detector.flag_outliers(normal_df, ["norm"], method="invalid")

    def test_flag_outliers_all_clean(self, clean_df):
        detector = OutlierDetector()
        df_result, summary = detector.flag_outliers(clean_df, ["x", "y"])
        assert summary["x"]["n_outliers"] == 0
        assert summary["y"]["n_outliers"] == 0

    def test_flag_outliers_missing_column_handled(self, normal_df):
        detector = OutlierDetector()
        # Include a non-existent column
        df_result, summary = detector.flag_outliers(normal_df, ["norm", "nonexistent"])
        assert "norm" in summary
        assert "error" in summary["nonexistent"]

    def test_flag_outliers_non_numeric_column_handled(self, normal_df):
        detector = OutlierDetector()
        df_result, summary = detector.flag_outliers(normal_df, ["categorical"])
        assert "error" in summary["categorical"]

    def test_flag_outliers_exact_outlier_count(self, normal_df):
        detector = OutlierDetector()
        # norm has 3 injected outliers: 100, -100, 15
        df_result, summary = detector.flag_outliers(normal_df, ["norm"])
        assert summary["norm"]["n_outliers"] >= 3

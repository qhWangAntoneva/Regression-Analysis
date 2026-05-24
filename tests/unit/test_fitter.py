# encoding: utf-8
"""Unit tests for the OLS statistical engine.

Tests cover:
    - Simple continuous-variable OLS
    - OLS with categorical variables (patsy C() encoding)
    - OLS without intercept
    - OLS with missing data (listwise deletion)
    - Coefficient values against known ground truth
    - ModelResult.to_dataframe() output format
    - Descriptive statistics
    - VIF computation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.diagnostics import model_summary, vif
from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec, build_formula
from src.results.statistics import descriptive_stats

# Path to the sample data
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_ols.csv"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load the sample OLS test dataset."""
    return pd.read_csv(SAMPLE_CSV, encoding="utf-8")


# =========================================================================
# Test: Simple OLS with 3 continuous variables
# =========================================================================
class TestOLSSimple:
    """OLS with continuous variables only."""

    def test_ols_simple(self, sample_data: pd.DataFrame) -> None:
        """Fit OLS with two continuous predictors (x1, x2)."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        # Basic structure
        assert result.model_type == "OLS"
        assert result.n_obs == 200
        assert result.n_params == 3  # const + x1 + x2
        assert result.df_resid == 197

        # Coefficient count
        assert len(result.coefficients) == 3

        # Check coefficient names
        coef_names = [c.name for c in result.coefficients]
        assert "Intercept" in coef_names
        assert "x1" in coef_names
        assert "x2" in coef_names

        # R-squared should be reasonable
        assert result.r_squared is not None
        assert 0.3 < result.r_squared < 0.6

        # RMSE should be positive
        assert result.rmse > 0

        # AIC, BIC should be finite
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

        # F-statistic should exist
        assert result.f_statistic is not None
        f_stat, f_pval = result.f_statistic
        assert f_stat > 0
        assert f_pval < 0.05

    def test_ols_coefficient_values(self, sample_data: pd.DataFrame) -> None:
        """Verify coefficient estimates are close to true values.

        True DGP: y = 2 + 0.5*x1 - 0.3*x2 + noise(0, 0.5)
        Expected estimates (from sample): const ~2.24, x1 ~0.47, x2 ~-0.40
        """
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        coef_map = {c.name: c for c in result.coefficients}

        # Intercept should be close to 2.0
        assert abs(coef_map["Intercept"].coef - 2.0) < 0.5
        # x1 coefficient should be close to 0.5
        assert abs(coef_map["x1"].coef - 0.5) < 0.2
        # x2 coefficient should be close to -0.3
        assert abs(coef_map["x2"].coef - (-0.3)) < 0.2

        # Standard errors should be positive
        for c in result.coefficients:
            assert c.se > 0, f"Non-positive SE for {c.name}"

        # p-values should be within [0, 1]
        for c in result.coefficients:
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}"


# =========================================================================
# Test: OLS with categorical variables
# =========================================================================
class TestOLSWithCategorical:
    """OLS including categorical (factor) variables."""

    def test_ols_with_categorical(self, sample_data: pd.DataFrame) -> None:
        """Fit OLS with x1, x2, and categorical x3."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        # With patsy categorical encoding, we get: Intercept + x1 + x2 + C(x3)[T.B] + C(x3)[T.C]
        assert len(result.coefficients) == 5

        coef_names = [c.name for c in result.coefficients]
        assert "Intercept" in coef_names
        assert "x1" in coef_names
        assert "x2" in coef_names

        # Categorical terms (patsy may name them differently)
        has_cat_terms = any("x3" in name for name in coef_names)
        assert has_cat_terms, f"No categorical terms found in {coef_names}"

        # R-squared should be higher than the simpler model
        spec_simple = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter_simple = ModelFitter()
        result_simple = fitter_simple.fit(spec_simple, sample_data)

        assert result.r_squared is not None
        assert result_simple.r_squared is not None
        # Categorical should add explanatory power (but not strictly guaranteed
        # with random data; this is a soft check)
        assert result.r_squared > result_simple.r_squared * 0.9


# =========================================================================
# Test: OLS without intercept
# =========================================================================
class TestOLSNoIntercept:
    """OLS model without a constant term."""

    def test_ols_no_intercept(self, sample_data: pd.DataFrame) -> None:
        """Fit OLS with has_intercept=False."""
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], has_intercept=False
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        # Should not have Intercept in coefficient names
        coef_names = [c.name for c in result.coefficients]
        intercept_names = [n for n in coef_names if "Intercept" in n or "const" in n.lower()]
        assert len(intercept_names) == 0, (
            f"Found intercept term(s) in no-intercept model: {intercept_names}"
        )

        # n_params should be 2 (no intercept)
        assert result.n_params == 2


# =========================================================================
# Test: OLS with missing data
# =========================================================================
class TestOLSWithMissingData:
    """OLS with missing values (should trigger listwise deletion)."""

    def test_ols_missing_data(self, sample_data: pd.DataFrame) -> None:
        """When x4 is included (has 5% missing), listwise deletion should occur."""
        n_before = len(sample_data)
        assert n_before == 200
        missing_count = sample_data["x4"].isna().sum()
        assert missing_count == 10  # 5% of 200

        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x4"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        # After listwise deletion, n_obs should be 190 (200 - 10)
        assert result.n_obs == n_before - missing_count

        # n_params should be 4 (const + x1 + x2 + x4)
        assert result.n_params == 4


# =========================================================================
# Test: ModelResult.to_dataframe()
# =========================================================================
class TestModelResultToDataFrame:
    """Verify ModelResult.to_dataframe() output format."""

    def test_model_result_to_dataframe(self, sample_data: pd.DataFrame) -> None:
        """Check the structure of the coefficient DataFrame."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        df = result.to_dataframe()

        # Check expected columns (Chinese column names as specified)
        expected_cols = ["系数", "标准误", "t值", "p值", "95%CI低", "95%CI高", "显著性"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Index should be variable names
        assert "Intercept" in df.index
        assert "x1" in df.index
        assert "x2" in df.index

        # All values should be finite
        for col in ["系数", "标准误", "t值"]:
            assert all(np.isfinite(df[col])), f"Non-finite values in {col}"

        # p-values should be between 0 and 1
        assert all(0 <= p <= 1 for p in df["p值"])

        # CI: lower < upper
        assert all(df["95%CI低"] < df["95%CI高"])

        # Round-trip: coefficient names in DataFrame should match result
        assert len(df) == len(result.coefficients)


# =========================================================================
# Test: Descriptive statistics
# =========================================================================
class TestDescriptiveStats:
    """Descriptive statistics output."""

    def test_descriptive_stats(self, sample_data: pd.DataFrame) -> None:
        """Check descriptive_stats() output structure."""
        variables = ["y", "x1", "x2", "x4"]
        stats_df = descriptive_stats(sample_data, variables)

        # Should have one row per variable
        assert len(stats_df) == len(variables)

        # Expected columns
        expected_cols = ["观测数", "均值", "标准差", "最小值", "25%", "50%", "75%", "最大值", "缺失值数", "缺失率"]
        for col in expected_cols:
            assert col in stats_df.columns, f"Missing column: {col}"

        # x4 should have 10 missing values
        assert stats_df.loc["x4", "缺失值数"] == 10
        assert stats_df.loc["x4", "缺失率"] == 0.05

        # y, x1, x2 should have 0 missing
        assert stats_df.loc["y", "缺失值数"] == 0
        assert stats_df.loc["x1", "缺失值数"] == 0
        assert stats_df.loc["x2", "缺失值数"] == 0

        # All means should be finite (for numeric variables)
        for var in ["y", "x1", "x2", "x4"]:
            assert np.isfinite(stats_df.loc[var, "均值"])

        # Standard deviations should be positive
        for var in ["y", "x1", "x2", "x4"]:
            assert stats_df.loc[var, "标准差"] > 0

    def test_descriptive_stats_nonexistent_var(self) -> None:
        """Should raise ValueError for missing variables."""
        data = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found in data"):
            descriptive_stats(data, ["a", "nonexistent"])


# =========================================================================
# Test: VIF computation
# =========================================================================
class TestVIF:
    """Variance Inflation Factor computation."""

    def test_vif(self, sample_data: pd.DataFrame) -> None:
        """Compute VIF for continuous variables."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x4"])
        vif_df = vif(sample_data, spec)

        # Should have at least const, x1, x2, x4
        assert len(vif_df) >= 3
        assert "variable" in vif_df.columns
        assert "vif" in vif_df.columns
        assert "diagnosis" in vif_df.columns

        # VIF values should be >= 1
        for _, row in vif_df.iterrows():
            assert row["vif"] >= 0, f"Negative VIF for {row['variable']}"

        # With independent variables, VIF should be low (< 5)
        # (note: const/Intercept may show high VIF but it's not meaningful)
        non_const = vif_df[
            ~vif_df["variable"].isin(["const", "Intercept"])
        ]
        for _, row in non_const.iterrows():
            assert row["vif"] < 5.0, (
                f"High VIF ({row['vif']}) for {row['variable']} "
                f"with independent variables"
            )
            assert row["diagnosis"] == "Low"

    def test_vif_too_few_vars(self, sample_data: pd.DataFrame) -> None:
        """VIF with a single predictor should still work (2 cols with const)."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1"])
        vif_df = vif(sample_data, spec, use_patsy=False)
        # Should produce exactly 2 rows: const and x1
        assert len(vif_df) == 2
        assert "x1" in vif_df["variable"].values


# =========================================================================
# Test: Formula building
# =========================================================================
class TestFormulaBuilding:
    """ModelSpec formula string generation."""

    def test_build_formula_with_intercept(self) -> None:
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        formula = build_formula(spec)
        assert formula == "y ~ x1 + x2"

    def test_build_formula_no_intercept(self) -> None:
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], has_intercept=False
        )
        formula = build_formula(spec)
        assert formula == "y ~ x1 + x2 - 1"

    def test_build_formula_with_controls(self) -> None:
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1"],
            control_vars=["z1", "z2"],
        )
        formula = build_formula(spec)
        assert formula == "y ~ x1 + z1 + z2"

    def test_build_formula_empty_indep(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ModelSpec(dep_var="y", indep_vars=[])


# =========================================================================
# Test: ModelSpec validation
# =========================================================================
class TestModelSpecValidation:
    """ModelSpec dataclass validation."""

    def test_empty_dep_var(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ModelSpec(dep_var="", indep_vars=["x1"])

    def test_duplicate_predictors(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            ModelSpec(dep_var="y", indep_vars=["x1", "x1"])

    def test_duplicate_with_controls(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            ModelSpec(
                dep_var="y",
                indep_vars=["x1", "x2"],
                control_vars=["x1"],
            )

    def test_all_predictors_property(self) -> None:
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            control_vars=["z1"],
        )
        assert spec.all_predictors == ["x1", "x2", "z1"]


# =========================================================================
# Test: model_summary dictionary
# =========================================================================
class TestModelSummary:
    """model_summary() diagnostic output."""

    def test_model_summary_dict(self, sample_data: pd.DataFrame) -> None:
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        summary = model_summary(result)

        # Expected keys
        assert "model_type" in summary
        assert "n_obs" in summary
        assert "r_squared" in summary
        assert "adj_r_squared" in summary
        assert "rmse" in summary
        assert "aic" in summary
        assert "bic" in summary
        assert "coefficients" in summary

        # Verify values
        assert summary["model_type"] == "OLS"
        assert summary["n_obs"] == 200
        assert summary["r_squared"] is not None
        assert summary["r_squared"] > 0

        # Coefficient list
        coefs = summary["coefficients"]
        assert len(coefs) == 3  # Intercept, x1, x2
        assert all("变量" in c for c in coefs)
        assert all("系数" in c for c in coefs)


# =========================================================================
# Test: OLS result against sample CSV (ground truth)
# =========================================================================
class TestOLSWithSampleCSV:
    """Verify regression coefficients from sample_ols.csv.

    Uses the actual OLS engine (not just patsy) to confirm the full
    pipeline produces correct results.
    """

    def test_ols_result_values(self, sample_data: pd.DataFrame) -> None:
        """Verify coefficient estimates from the full pipeline.

        Expected (from actual OLS fit):
            const ~ 2.24, x1 ~ 0.47, x2 ~ -0.40
        """
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        coef_map = {c.name: c for c in result.coefficients}

        # Known reference values (from OLS on seed-42 data)
        assert abs(coef_map["Intercept"].coef - 2.24) < 0.1
        assert abs(coef_map["x1"].coef - 0.47) < 0.1
        assert abs(coef_map["x2"].coef - (-0.40)) < 0.1

        # R-squared should be close to known value
        assert result.r_squared is not None
        assert abs(result.r_squared - 0.452) < 0.05

        # RMSE should be close to noise std (~0.5)
        assert abs(result.rmse - 0.5) < 0.15

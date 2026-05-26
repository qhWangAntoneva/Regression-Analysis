"""Integration tests for the full OLS regression workflow.

Tests the complete pipeline from data loading through model fitting
to result extraction and formatting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_engine import run_ols
from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.results.table import CoefficientRow, ModelResult

# Paths
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_ols.csv"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load the sample OLS test dataset."""
    return pd.read_csv(SAMPLE_CSV, encoding="utf-8")


# =========================================================================
# Test: End-to-end OLS workflow
# =========================================================================
class TestEndToEndOLS:
    """Complete end-to-end OLS workflow integration test."""

    def test_end_to_end_ols(self, sample_data: pd.DataFrame) -> None:
        """Test the full pipeline: load CSV -> ModelSpec -> run_ols -> ModelResult -> to_dataframe.

        Validates every step of the pipeline produces consistent,
        well-formed output.
        """
        # --- Step 1: Build specification ---
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        assert spec.dep_var == "y"
        assert len(spec.indep_vars) == 3
        assert spec.has_intercept is True

        # --- Step 2: Run OLS via the engine directly ---
        result = run_ols(sample_data, spec)

        # --- Step 3: Verify ModelResult structure ---
        assert isinstance(result, ModelResult)
        assert result.model_type == "OLS"
        assert result.method == "OLS"
        assert result.dep_var == "y"
        assert result.n_obs == 200
        assert result.n_params == 5  # Intercept + x1 + x2 + C(x3)[T.B] + C(x3)[T.C]

        # --- Step 4: Verify coefficients ---
        assert len(result.coefficients) == result.n_params

        # All coefficients should have valid statistics
        for c in result.coefficients:
            assert isinstance(c, CoefficientRow)
            assert isinstance(c.name, str) and len(c.name) > 0
            assert np.isfinite(c.coef), f"Non-finite coef for {c.name}"
            assert c.se > 0, f"Non-positive SE for {c.name}"
            assert np.isfinite(c.t_stat), f"Non-finite t-stat for {c.name}"
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}"
            assert c.ci_lower < c.ci_upper, (
                f"CI inversion for {c.name}: "
                f"lower={c.ci_lower} >= upper={c.ci_upper}"
            )

        # --- Step 5: Verify model-level statistics ---
        assert result.r_squared is not None
        assert result.adj_r_squared is not None
        assert result.r_squared > 0.3
        assert result.adj_r_squared > 0.3
        assert result.rmse > 0
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

        assert result.f_statistic is not None
        f_stat, f_pval = result.f_statistic
        assert f_stat > 0
        assert f_pval < 0.05

        # Log-likelihood should be finite
        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)

        # --- Step 6: Verify to_dataframe() output ---
        coef_df = result.to_dataframe()
        assert isinstance(coef_df, pd.DataFrame)
        assert len(coef_df) == result.n_params

        expected_cols = ["系数", "标准误", "t值", "p值", "95%CI低", "95%CI高", "显著性"]
        for col in expected_cols:
            assert col in coef_df.columns

        # Index should contain 'Intercept'
        assert "Intercept" in coef_df.index

        # All coefficients should be finite in the DataFrame
        for col in ["系数", "标准误", "t值"]:
            assert all(np.isfinite(coef_df[col])), f"Non-finite values in {col}"

        # CI consistency in DataFrame
        assert all(coef_df["95%CI低"] < coef_df["95%CI高"])

        # --- Step 7: Verify significance stars ---
        # Intercept should be highly significant
        if "Intercept" in coef_df.index:
            assert coef_df.loc["Intercept", "显著性"] != ""

        # --- Step 8: Verify summary string ---
        summary_str = result.summary()
        assert "OLS Regression Results" in summary_str
        assert result.dep_var in summary_str
        assert str(result.n_obs) in summary_str
        assert "R-squared" in summary_str
        assert "Significance" in summary_str
        assert "***" in summary_str or "**" in summary_str or "*" in summary_str

    def test_end_to_end_no_intercept(self, sample_data: pd.DataFrame) -> None:
        """End-to-end test for OLS without intercept."""
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], has_intercept=False
        )
        result = run_ols(sample_data, spec)

        assert result.n_params == 2  # x1, x2 only
        assert len(result.coefficients) == 2

        # No term named "Intercept"
        coef_names = [c.name for c in result.coefficients]
        assert not any("Intercept" in n for n in coef_names)

        # All coefficients should still be valid
        for c in result.coefficients:
            assert c.se > 0

    def test_end_to_end_missing_data(self, sample_data: pd.DataFrame) -> None:
        """End-to-end test with missing data (listwise deletion)."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x4"])
        result = run_ols(sample_data, spec)

        # 10 out of 200 rows have missing x4 -> 190 obs
        assert result.n_obs == 190


# =========================================================================
# Test: Multiple model specifications
# =========================================================================
class TestMultipleSpecs:
    """Fitting multiple model specifications for comparison."""

    def test_multiple_specs(self, sample_data: pd.DataFrame) -> None:
        """Fit two model specs and compare results.

        Model 1: y ~ x1
        Model 2: y ~ x1 + x2

        Model 2 should have higher R-squared (or at least non-decreasing).
        """
        spec1 = ModelSpec(dep_var="y", indep_vars=["x1"])
        spec2 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])

        fitter = ModelFitter()
        results = fitter.fit_multiple([spec1, spec2], sample_data)

        assert len(results) == 2

        # Both should be valid
        for i, result in enumerate(results):
            assert result.dep_var == "y"
            assert result.r_squared is not None
            assert result.r_squared > 0
            assert result.rmse > 0

        # Model 2 (with x2) should have higher or equal R-squared
        # (R-squared is non-decreasing as predictors are added)
        assert results[1].r_squared is not None
        assert results[0].r_squared is not None
        assert results[1].r_squared >= results[0].r_squared

        # RMSE should be lower for the richer model
        assert results[1].rmse <= results[0].rmse

        # Each model should have correct number of parameters
        assert results[0].n_params == 2  # Intercept + x1
        assert results[1].n_params == 3  # Intercept + x1 + x2

    def test_multiple_specs_nested_comparison(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Fit and compare three nested models.

        M1: y ~ x1
        M2: y ~ x1 + x2
        M3: y ~ x1 + x2 + x4
        """
        specs = [
            ModelSpec(dep_var="y", indep_vars=["x1"]),
            ModelSpec(dep_var="y", indep_vars=["x1", "x2"]),
            ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x4"]),
        ]

        fitter = ModelFitter()
        results = fitter.fit_multiple(specs, sample_data)

        assert len(results) == 3

        # Verify monotonic R-squared progression
        r2s = []
        for r in results:
            assert r.r_squared is not None
            r2s.append(r.r_squared)

        for i in range(1, len(r2s)):
            assert r2s[i] >= r2s[i - 1], (
                f"R-squared decreased from model {i} to {i+1}: "
                f"{r2s[i-1]:.4f} -> {r2s[i]:.4f}"
            )

        # Model 3 uses x4 which has missing values -> fewer obs
        assert results[0].n_obs == 200
        assert results[1].n_obs == 200
        assert results[2].n_obs == 190  # 10 missing in x4

        # Test fitted_results property
        cached = fitter.fitted_results
        assert len(cached) == 3

        # Test clear
        fitter.clear()
        assert len(fitter.fitted_results) == 0

    def test_coefficient_consistency_across_specs(
        self, sample_data: pd.DataFrame
    ) -> None:
        """When adding controls, coefficients should change gracefully.

        Adding x2 to a model with x1 should change the x1 coefficient
        modestly (not flip sign or explode).
        """
        spec1 = ModelSpec(dep_var="y", indep_vars=["x1"])
        spec2 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])

        fitter = ModelFitter()
        results = fitter.fit_multiple([spec1, spec2], sample_data)

        # Get x1 coefficient from both models
        coef1 = next(c for c in results[0].coefficients if c.name == "x1")
        coef2 = next(c for c in results[1].coefficients if c.name == "x1")

        # Sign should be the same (both positive)
        assert coef1.coef > 0
        assert coef2.coef > 0

        # Change should be modest (within 30%)
        change_pct = abs(coef2.coef - coef1.coef) / abs(coef1.coef)
        assert change_pct < 0.3, (
            f"x1 coefficient changed by {change_pct*100:.1f}% "
            f"when adding x2: {coef1.coef:.4f} -> {coef2.coef:.4f}"
        )


# =========================================================================
# Test: Data loading and preprocessing
# =========================================================================
class TestDataLoading:
    """Verify that the sample data loads correctly."""

    def test_sample_data_shape(self, sample_data: pd.DataFrame) -> None:
        """Sample CSV should have correct shape."""
        assert sample_data.shape == (200, 8)

    def test_sample_data_columns(self, sample_data: pd.DataFrame) -> None:
        """Should have all expected columns."""
        expected = {"y", "x1", "x2", "x3", "x3_B", "x3_C", "x4", "cat1"}
        assert set(sample_data.columns) == expected

    def test_sample_data_missing(self, sample_data: pd.DataFrame) -> None:
        """x4 should have 10 missing values."""
        assert sample_data["x4"].isna().sum() == 10

    def test_sample_data_categories(self, sample_data: pd.DataFrame) -> None:
        """x3 should have categories A, B, C."""
        assert set(sample_data["x3"].unique()) == {"A", "B", "C"}


# =========================================================================
# Test: Summary string output
# =========================================================================
class TestModelSummaryString:
    """Human-readable model summary output."""

    def test_summary_contains_key_info(self, sample_data: pd.DataFrame) -> None:
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        result = run_ols(sample_data, spec)
        summary = result.summary()

        # Should contain key model details
        assert "OLS Regression Results" in summary
        assert "Dependent Variable:" in summary
        assert "R-squared:" in summary
        assert "Adj. R-squared:" in summary
        assert "RMSE:" in summary
        assert "F-statistic:" in summary
        assert "AIC:" in summary
        assert "BIC:" in summary
        assert "Coefficient" in summary or "Variable" in summary
        assert "Significance:" in summary

        # Should contain variable names
        assert "Intercept" in summary
        assert "x1" in summary
        assert "x2" in summary

"""Unit tests for the MixedLM (multilevel) regression engine.

Tests cover:
    - Basic fitting on grouped data
    - Fixed effects coefficients are reasonable
    - Standard errors > 0
    - Random effects variance > 0 (group-level variation exists)
    - Group count is correct
    - Group variable metadata preserved
    - Error handling: missing group_var
    - Error handling: single group (no variation)
    - Test statistics present
    - R-squared-like measures present
    - Variable labels preserved
    - P-values in [0, 1]
    - CI ordering (lower < upper)
    - Convergence flag
    - RE variance dictionary content
    - Multiple predictors
    - No-intercept model
    - ModelResult type is correct
    - to_dataframe() works
    - to_summary_dict() works
    - Summary text format
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_mixedlm_engine import (
    extract_mixedlm,
    run_and_extract_mixedlm,
    run_mixedlm,
)
from src.modeling.specification import ModelSpec

# =========================================================================
# Helpers
# =========================================================================


def make_grouped_data(
    n_groups: int = 20,
    n_per_group: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic dataset with group-level random effects.

    DGP: y = 2.0 + 1.5*x1 - 0.8*x2 + group_effect + noise
    group_effect ~ N(0, 0.5)
    noise ~ N(0, 0.3)
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x1 = rng.normal(0, 1, n_groups * n_per_group)
    x2 = rng.normal(0, 1, n_groups * n_per_group)
    group_effects = rng.normal(0, 0.5, n_groups)
    y = (
        2.0
        + 1.5 * x1
        - 0.8 * x2
        + group_effects[groups]
        + rng.normal(0, 0.3, n_groups * n_per_group)
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})


def make_tiny_grouped_data(
    n_groups: int = 6,
    n_per_group: int = 8,
    seed: int = 99,
) -> pd.DataFrame:
    """Create a small grouped dataset (fewer groups, faster fit)."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x1 = rng.normal(0, 1, n_groups * n_per_group)
    group_effects = rng.normal(0, 0.4, n_groups)
    y = 1.0 + 0.7 * x1 + group_effects[groups] + rng.normal(0, 0.25, n_groups * n_per_group)
    return pd.DataFrame({"y": y, "x1": x1, "group": groups})


# =========================================================================
# Tests: Basic fitting
# =========================================================================


class TestMixedLMBasic:
    """Basic MixedLM fitting on grouped data."""

    def test_fit_success(self) -> None:
        """MixedLM should fit without error."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)
        assert result.model_type == "mixedlm"
        assert result.n_obs == 300
        assert result.n_params == 3  # Intercept + x1 + x2

    def test_coefficients_reasonable(self) -> None:
        """FE coefficients should be close to DGP values."""
        df = make_grouped_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        coef_map = {c.name: c for c in result.coefficients}
        # DGP: intercept=2.0, x1=1.5, x2=-0.8
        assert abs(coef_map["Intercept"].coef - 2.0) < 0.5, "Intercept off by >0.5"
        assert abs(coef_map["x1"].coef - 1.5) < 0.2, "x1 coef off by >0.2"
        assert abs(coef_map["x2"].coef - (-0.8)) < 0.2, "x2 coef off by >0.2"

    def test_all_standard_errors_positive(self) -> None:
        """All standard errors should be > 0."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        for c in result.coefficients:
            assert c.se > 0, f"Non-positive SE for {c.name}"

    def test_pvalues_in_range(self) -> None:
        """All p-values should be in [0, 1]."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        for c in result.coefficients:
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}: {c.pvalue}"

    def test_ci_ordering(self) -> None:
        """CI lower should be less than CI upper for all coefficients."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        for c in result.coefficients:
            assert c.ci_lower < c.ci_upper, f"CI order wrong for {c.name}"


# =========================================================================
# Tests: Random effects
# =========================================================================


class TestRandomEffects:
    """Random effects variance components."""

    def test_re_variance_positive(self) -> None:
        """Group-level variation should yield positive RE variance."""
        df = make_grouped_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert len(result.re_var) > 0, "No RE variance components"
        for name, val in result.re_var.items():
            assert val > 0, f"RE variance for '{name}' should be >0, got {val}"

    def test_re_var_is_dict(self) -> None:
        """re_var should be a dict mapping name to float."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert isinstance(result.re_var, dict)
        for k, v in result.re_var.items():
            assert isinstance(k, str)
            assert isinstance(v, float)
            assert v > 0

    def test_re_var_has_group_var_key(self) -> None:
        """RE variance dict should contain 'Group Var' key."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert "Group Var" in result.re_var


# =========================================================================
# Tests: Group metadata
# =========================================================================


class TestGroupMetadata:
    """Group variable metadata preservation."""

    def test_group_count_correct(self) -> None:
        """group_count should match the number of unique groups."""
        df = make_grouped_data(n_groups=20, n_per_group=15)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.group_count == 20

    def test_group_var_preserved(self) -> None:
        """group_var attribute should match the specification."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "region"
        df["region"] = df["group"]  # rename

        result = run_and_extract_mixedlm(df, spec)

        assert result.group_var == "region"

    def test_group_count_fewer_groups(self) -> None:
        """Group count correct with fewer groups."""
        df = make_grouped_data(n_groups=8, n_per_group=10)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.group_count == 8

    def test_converged_flag_true(self) -> None:
        """Model should report converged=True for well-behaved data."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.mixedlm_converged is True

    def test_scale_positive(self) -> None:
        """Residual scale should be positive."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.mixedlm_scale is not None
        assert result.mixedlm_scale > 0


# =========================================================================
# Tests: Error handling
# =========================================================================


class TestErrorHandling:
    """MixedLM error handling."""

    def test_missing_group_var_raises(self) -> None:
        """Should raise when spec has no group_var."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")

        with pytest.raises(ValueError, match="group_var"):
            run_mixedlm(df, spec)

    def test_group_var_not_in_data_raises(self) -> None:
        """Should raise when group column is not in DataFrame."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "nonexistent_column"

        with pytest.raises(ValueError, match="not found"):
            run_mixedlm(df, spec)

    def test_single_group_raises(self) -> None:
        """Should raise when there is only 1 group."""
        df = make_grouped_data(n_groups=1, n_per_group=25)
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"

        with pytest.raises(ValueError, match="at least 2 groups"):
            run_mixedlm(df, spec)


# =========================================================================
# Tests: R-squared-like measures
# =========================================================================


class TestRSquaredMeasures:
    """R-squared-like measures for MixedLM."""

    def test_r_squared_present_and_positive(self) -> None:
        """Conditional R-squared should be present and > 0."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.r_squared is not None
        assert 0 < result.r_squared < 1.0

    def test_adj_r_squared_present(self) -> None:
        """Adjusted R-squared should be present."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.adj_r_squared is not None
        assert 0 < result.adj_r_squared < 1.0

    def test_adj_r2_less_than_r2(self) -> None:
        """Adjusted R² should be <= R²."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.adj_r_squared <= result.r_squared

    def test_rmse_positive(self) -> None:
        """RMSE should be positive."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.rmse is not None
        assert result.rmse > 0


# =========================================================================
# Tests: Test statistics
# =========================================================================


class TestStatistics:
    """Test statistics on coefficient rows."""

    def test_t_stat_available(self) -> None:
        """t_stat should be available on coefficient rows."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        for c in result.coefficients:
            assert isinstance(c.t_stat, float)
            assert np.isfinite(c.t_stat)

    def test_significance_stars_set(self) -> None:
        """Significance stars should be auto-computed."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        coef_map = {c.name: c for c in result.coefficients}
        # x1 and x2 are strong predictors => should be ***
        assert coef_map["x1"].significance == "***"
        assert coef_map["x2"].significance == "***"

    def test_log_likelihood_finite(self) -> None:
        """Log-likelihood should be finite."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)
        # Log-likelihood should be negative (typical for REML)
        assert result.log_likelihood < 0


# =========================================================================
# Tests: Variable labels
# =========================================================================


class TestVariableLabels:
    """Variable label preservation."""

    def test_variable_labels_preserved(self) -> None:
        """Variable labels should be in the result."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert len(result.variable_labels) > 0
        assert "Intercept" in result.variable_labels
        assert result.variable_labels["Intercept"] == "Intercept"


# =========================================================================
# Tests: Data output
# =========================================================================


class TestDataOutput:
    """to_dataframe() and to_summary_dict() for MixedLM."""

    def test_to_dataframe_uses_t_value_column(self) -> None:
        """Non-MLE models use 't值' column (MixedLM is not MLE)."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        out = result.to_dataframe()
        assert "t值" in out.columns, f"Expected 't值', got {list(out.columns)}"

    def test_to_summary_dict(self) -> None:
        """to_summary_dict() should work and contain mixedlm-specific keys."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        d = result.to_summary_dict()
        assert d["model_type"] == "mixedlm"
        assert d["method"] == "MixedLM (REML)"
        assert d["r_squared"] is not None
        assert d["f_statistic"] is None

    def test_summary_text_format(self) -> None:
        """summary() should mention MixedLM and REML."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        text = result.summary()
        assert "MixedLM (REML)" in text
        assert "R-squared" in text
        assert "p>|t|" in text

    def test_to_latex_row(self) -> None:
        """to_latex_row() should produce valid LaTeX.

        Non-MLE models produce 8 parts:
        dep_var & n & r2 & adj_r2 & f & fp & aic & bic.
        For MixedLM: f_statistic is None (shows "N/A") and AIC/BIC
        are NaN for REML (shows "nan").
        """
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        assert len(parts) == 8, f"Expected 8 parts, got {len(parts)}: {parts}"
        # first three parts should have valid values
        assert parts[0] == "y"
        assert parts[1] == str(result.n_obs)


# =========================================================================
# Tests: Multiple predictors and special specs
# =========================================================================


class TestMultiplePredictors:
    """MixedLM with various predictor configurations."""

    def test_single_predictor(self) -> None:
        """MixedLM with just one predictor."""
        df = make_tiny_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert len(result.coefficients) == 2  # Intercept + x1
        assert result.r_squared is not None
        assert result.r_squared > 0

    def test_multiple_predictors(self) -> None:
        """MixedLM with three predictors."""
        rng = np.random.default_rng(123)
        n_groups, n_per = 10, 12
        groups = np.repeat(np.arange(n_groups), n_per)
        x1, x2, x3 = rng.normal(0, 1, (3, n_groups * n_per))
        g_eff = rng.normal(0, 0.4, n_groups)
        y = 1.5 + 0.6 * x1 - 0.4 * x2 + 0.3 * x3 + g_eff[groups] + rng.normal(0, 0.2, n_groups * n_per)  # noqa: E501
        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "group": groups})

        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(data, spec)

        assert len(result.coefficients) == 4  # Intercept + 3 predictors
        assert result.n_params == 4

    def test_no_intercept(self) -> None:
        """MixedLM without intercept."""
        rng = np.random.default_rng(77)
        n_groups, n_per = 8, 10
        groups = np.repeat(np.arange(n_groups), n_per)
        x1 = rng.normal(0, 1, n_groups * n_per)
        g_eff = rng.normal(0, 0.3, n_groups)
        y = 0.7 * x1 + g_eff[groups] + rng.normal(0, 0.2, n_groups * n_per)
        data = pd.DataFrame({"y": y, "x1": x1, "group": groups})

        spec = ModelSpec(dep_var="y", indep_vars=["x1"], has_intercept=False, model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(data, spec)

        assert len(result.coefficients) == 1  # x1 only, no Intercept
        coef_names = [c.name for c in result.coefficients]
        assert "Intercept" not in coef_names
        assert "x1" in coef_names

    def test_with_control_vars(self) -> None:
        """MixedLM with independent + control variables."""
        df = make_grouped_data()
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1"],
            control_vars=["x2"],
            model_type="mixedlm",
        )
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        coef_names = [c.name for c in result.coefficients]
        assert "x1" in coef_names
        assert "x2" in coef_names
        assert len(result.coefficients) == 3  # Intercept + x1 + x2


# =========================================================================
# Tests: run_mixedlm + extract_mixedlm split API
# =========================================================================


class TestSplitAPI:
    """run_mixedlm() -> extract_mixedlm() pipeline."""

    def test_split_pipeline(self) -> None:
        """run_mixedlm + extract_mixedlm should give same result."""
        df = make_tiny_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"

        fitted, labels = run_mixedlm(df, spec)
        result = extract_mixedlm(fitted, dep_var="y", variable_labels=labels)

        assert result.model_type == "mixedlm"
        assert result.n_obs == len(df)
        assert len(result.coefficients) == 2  # Intercept + x1
        assert result.group_count == 6

    def test_split_pipeline_group_metadata(self) -> None:
        """Group metadata should be accessible after split pipeline."""
        df = make_tiny_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"

        fitted, labels = run_mixedlm(df, spec)
        result = extract_mixedlm(fitted, variable_labels=labels)

        assert result.group_var == "group"
        assert result.group_count == 6
        assert len(result.re_var) > 0


# =========================================================================
# Tests: Large-ish dataset
# =========================================================================


class TestLargeData:
    """MixedLM on moderately large grouped data."""

    def test_large_groups(self) -> None:
        """Fit on 50 groups with 20 obs each (1000 total)."""
        rng = np.random.default_rng(555)
        n_groups, n_per = 50, 20
        groups = np.repeat(np.arange(n_groups), n_per)
        x1 = rng.normal(0, 1, n_groups * n_per)
        g_eff = rng.normal(0, 0.3, n_groups)
        y = 0.5 + 1.2 * x1 + g_eff[groups] + rng.normal(0, 0.4, n_groups * n_per)
        data = pd.DataFrame({"y": y, "x1": x1, "group": groups})

        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(data, spec)

        assert result.n_obs == 1000
        assert result.group_count == 50
        assert result.r_squared is not None
        assert result.r_squared > 0.5


# =========================================================================
# Tests: ModelResult properties
# =========================================================================


class TestModelResultProperties:
    """ModelResult semantic properties for mixedlm."""

    def test_is_mle_model_false(self) -> None:
        """MixedLM is NOT an MLE model (uses REML)."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.is_mle_model is False

    def test_is_binary_choice_false(self) -> None:
        """MixedLM is not a binary choice model."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.is_binary_choice is False

    def test_is_count_model_false(self) -> None:
        """MixedLM is not a count model."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.is_count_model is False

    def test_f_statistic_none(self) -> None:
        """F-statistic is not available for MixedLM (no SS decomposition)."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        assert result.f_statistic is None


# =========================================================================
# Tests: ANOVA table
# =========================================================================


class TestAnovaTable:
    """anova_table() for MixedLM."""

    def test_anova_table_not_empty(self) -> None:
        """ANOVA table should NOT be empty for MixedLM (non-MLE model)."""
        df = make_tiny_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        anova = result.anova_table()
        assert not anova.empty
        assert "来源" in anova.columns
        assert len(anova) == 3  # Explained, Residual, Total
        assert anova.iloc[0]["来源"] == "回归(Explained)"
        assert anova.iloc[1]["来源"] == "残差(Residual)"
        assert anova.iloc[2]["来源"] == "总计(Total)"

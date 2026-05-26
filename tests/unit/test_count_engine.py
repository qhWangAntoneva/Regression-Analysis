# encoding: utf-8
"""Unit tests for the Count regression engines (Poisson and NegativeBinomial).

Tests cover:
    - Poisson: basic fitting, coefficient signs, SE > 0, p-values in [0,1]
    - Poisson: pseudo R-squared, z-statistics, IRR = exp(coef)
    - Poisson: OLS fields are None, LLR, log-likelihood, AIC/BIC
    - NegBin: basic fitting, dispersion > 0, IRR = exp(coef)
    - NegBin: overdispersion detection (NegBin fits better when Var > Mean)
    - Error handling: non-integer DV, negative DV values
    - Variable labels, to_dataframe(), to_summary_dict(), summary()
    - CI bounds correct, model_type stored correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.modeling.engines.statsmodels_count_engine import (
    extract_count_model,
    run_count_model,
)
from src.modeling.specification import ModelSpec
from src.results.table import _significance_stars, CoefficientRow, ModelResult


# =========================================================================
# Helpers: generate count data
# =========================================================================

def make_poisson_data(
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create count data from a Poisson DGP.

    DGP: lambda = exp(1.0 + 0.3*x1 - 0.15*x2), y ~ Poisson(lambda)
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lam = np.exp(1.0 + 0.3 * x1 - 0.15 * x2)
    y = rng.poisson(lam)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def make_overdispersed_data(
    n: int = 300,
    seed: int = 77,
) -> pd.DataFrame:
    """Create overdispersed count data (variance > mean).

    Uses a NegativeBinomial DGP to generate data where
    Var(y) > E(y).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 1.0 + 0.5 * x1 - 0.3 * x2
    mu = np.exp(eta)
    # Negative binomial with size=1 (geometric-like, heavy overdispersion)
    size = 1.0
    prob = size / (size + mu)
    y = rng.negative_binomial(size, prob)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# =========================================================================
# Tests: Poisson basic fitting
# =========================================================================

class TestPoissonBasic:
    """Basic Poisson regression on synthetic count data."""

    def test_poisson_fit_success(self) -> None:
        """Poisson should fit without error and return results."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        assert result.model_type == "poisson"
        assert result.method == "Poisson"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3  # Intercept + x1 + x2

    def test_poisson_coefficient_signs(self) -> None:
        """Coefficients should have expected signs from DGP."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        coef_map = {c.name: c for c in result.coefficients}
        # x1 has positive effect in DGP
        assert coef_map["x1"].coef > 0, "x1 should have positive coefficient"
        # x2 has negative effect in DGP
        assert coef_map["x2"].coef < 0, "x2 should have negative coefficient"

    def test_poisson_standard_errors_positive(self) -> None:
        """All standard errors should be positive."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert c.se > 0, f"Non-positive SE for {c.name}"

    def test_poisson_pvalues_in_range(self) -> None:
        """All p-values should be between 0 and 1."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}"


# =========================================================================
# Tests: Poisson model-level statistics
# =========================================================================

class TestPoissonModelStats:
    """Poisson model-level statistics."""

    def test_poisson_pseudo_r_squared(self) -> None:
        """Pseudo R-squared should be in [0, 1]."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.pseudo_r_squared is not None
        assert 0 <= result.pseudo_r_squared <= 1, (
            f"Pseudo R² = {result.pseudo_r_squared}, expected [0, 1]"
        )

    def test_poisson_ols_fields_are_none(self) -> None:
        """OLS-specific fields should be None for Poisson."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.r_squared is None
        assert result.adj_r_squared is None
        assert result.f_statistic is None
        assert result.rmse is None

    def test_poisson_log_likelihood_finite(self) -> None:
        """Log-likelihood should be finite and negative."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)
        assert result.log_likelihood < 0  # LL is typically negative

    def test_poisson_aic_bic_finite(self) -> None:
        """AIC and BIC should be finite."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert result.aic > 0
        # BIC may be positive or negative depending on the statsmodels version
        # (deviance-based vs LLF-based). Just verify it is finite.
        assert result.bic != 0, "BIC should not be zero"

    def test_poisson_llr(self) -> None:
        """LLR should be computed for Poisson models."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.llr is not None
        assert result.llr > 0, "LLR should be positive for a reasonable model"
        if result.llr_pvalue is not None:
            assert 0 <= result.llr_pvalue <= 1


# =========================================================================
# Tests: z-statistics for Poisson
# =========================================================================

class TestPoissonZStatistics:
    """z-statistics from Poisson model."""

    def test_z_stat_available(self) -> None:
        """z-statistics should be accessible on coefficient rows."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert c.z_stat == c.t_stat, "z_stat should equal t_stat for Poisson"

    def test_z_stat_nonzero(self) -> None:
        """z-statistics should be non-zero for predictive variables."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        coef_map = {c.name: c for c in result.coefficients}
        assert abs(coef_map["x1"].z_stat) > 0.1


# =========================================================================
# Tests: CI bounds correct
# =========================================================================

class TestPoissonCI:
    """Confidence interval bounds for Poisson models."""

    def test_ci_lower_less_than_upper(self) -> None:
        """CI lower bound should be less than upper bound."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert c.ci_lower < c.ci_upper, (
                f"CI inversion for {c.name}: "
                f"lower={c.ci_lower} >= upper={c.ci_upper}"
            )

    def test_ci_contains_coefficient(self) -> None:
        """95% CI should contain the estimated coefficient."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert c.ci_lower <= c.coef <= c.ci_upper, (
                f"CI [{c.ci_lower:.4f}, {c.ci_upper:.4f}] does not "
                f"contain coefficient {c.coef:.4f} for {c.name}"
            )


# =========================================================================
# Tests: IRR = exp(coef)
# =========================================================================

class TestPoissonIRR:
    """Incidence rate ratio: IRR = exp(coef)."""

    def test_irr_from_coefficients(self) -> None:
        """IRR = exp(coef) should be finite and positive."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            irr_val = np.exp(c.coef)
            assert np.isfinite(irr_val), f"Non-finite IRR for {c.name}"
            assert irr_val > 0, f"IRR should be positive for {c.name}"

    def test_irr_above_1_for_positive_coef(self) -> None:
        """IRR > 1 when coefficient is positive."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        coef_map = {c.name: c for c in result.coefficients}
        assert np.exp(coef_map["x1"].coef) > 1, "IRR for x1 (>0) should be >1"

    def test_irr_below_1_for_negative_coef(self) -> None:
        """IRR < 1 when coefficient is negative."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        coef_map = {c.name: c for c in result.coefficients}
        assert np.exp(coef_map["x2"].coef) < 1, "IRR for x2 (<0) should be <1"


# =========================================================================
# Tests: NegativeBinomial basic fitting
# =========================================================================

class TestNegBinBasic:
    """Negative Binomial regression on count data."""

    def test_negbin_fit_success(self) -> None:
        """NegativeBinomial should fit without error."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        assert result.model_type == "negbin"
        assert result.method == "NegativeBinomial"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3  # Intercept + x1 + x2

    def test_negbin_dispersion_positive(self) -> None:
        """Dispersion (scale) parameter should be > 0."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)

        # Dispersion is extracted from fitted_model.scale
        # Even with default GLM, scale should be set
        scale = getattr(fitted, "scale", None)
        assert scale is not None, "NB model should have a scale attribute"
        assert scale > 0, f"Expected positive dispersion, got {scale}"

    def test_negbin_standard_errors_positive(self) -> None:
        """All standard errors should be positive."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert c.se > 0, f"Non-positive SE for {c.name}"

    def test_negbin_pvalues_in_range(self) -> None:
        """All p-values should be between 0 and 1."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}"

    def test_negbin_irr_exp_coef(self) -> None:
        """IRR = exp(coef) should work for NB."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            irr_val = np.exp(c.coef)
            assert np.isfinite(irr_val)
            assert irr_val > 0

    def test_negbin_pseudo_r_squared(self) -> None:
        """Pseudo R-squared should be in [0, 1] for NegBin."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.pseudo_r_squared is not None
        assert 0 <= result.pseudo_r_squared <= 1, (
            f"Pseudo R² = {result.pseudo_r_squared}, expected [0, 1]"
        )

    def test_negbin_ols_fields_are_none(self) -> None:
        """OLS-specific fields should be None for NB."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.r_squared is None
        assert result.adj_r_squared is None
        assert result.f_statistic is None
        assert result.rmse is None


# =========================================================================
# Tests: Overdispersion — NegBin fits better than Poisson when Var > Mean
# =========================================================================

class TestOverdispersion:
    """When data is overdispersed (variance > mean), NegBin should fit better."""

    def test_overdispersed_data_has_var_gt_mean(self) -> None:
        """Verify that our overdispersed data has Var(y) > Mean(y)."""
        data = make_overdispersed_data(seed=77)
        assert data["y"].var() > data["y"].mean(), (
            f"Expected overdispersion: Var={data['y'].var():.2f}, "
            f"Mean={data['y'].mean():.2f}"
        )

    def test_negbin_better_aic_than_poisson_for_overdispersed(self) -> None:
        """NegBin should have lower AIC than Poisson on overdispersed data."""
        data = make_overdispersed_data(seed=77)

        poisson_spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        negbin_spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )

        poisson_fitted, _ = run_count_model(data, poisson_spec)
        negbin_fitted, _ = run_count_model(data, negbin_spec)

        poisson_result = extract_count_model(poisson_fitted)
        negbin_result = extract_count_model(negbin_fitted)

        # NegBin should have lower (better) AIC on overdispersed data
        assert negbin_result.aic < poisson_result.aic, (
            f"NegBin AIC ({negbin_result.aic:.1f}) should be lower than "
            f"Poisson AIC ({poisson_result.aic:.1f}) on overdispersed data"
        )

    def test_negbin_lower_bic_than_poisson_for_overdispersed(self) -> None:
        """NegBin should have lower BIC than Poisson on overdispersed data."""
        data = make_overdispersed_data(seed=77)

        poisson_spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        negbin_spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )

        poisson_fitted, _ = run_count_model(data, poisson_spec)
        negbin_fitted, _ = run_count_model(data, negbin_spec)

        poisson_result = extract_count_model(poisson_fitted)
        negbin_result = extract_count_model(negbin_fitted)

        assert negbin_result.bic < poisson_result.bic, (
            f"NegBin BIC ({negbin_result.bic:.1f}) should be lower than "
            f"Poisson BIC ({poisson_result.bic:.1f}) on overdispersed data"
        )


# =========================================================================
# Tests: Error handling
# =========================================================================

class TestCountModelValidation:
    """Input validation for count models."""

    def test_poisson_rejects_negative_dv(self) -> None:
        """Poisson should reject negative DV values."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        y = rng.choice([-1, 0, 1, 2], n)  # some negative values
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="poisson")

        with pytest.raises(ValueError, match="negative"):
            run_count_model(data, spec)

    def test_negbin_rejects_negative_dv(self) -> None:
        """NegativeBinomial should reject negative DV values."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        y = rng.choice([-1, 0, 1, 2], n)
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="negbin")

        with pytest.raises(ValueError, match="negative"):
            run_count_model(data, spec)

    def test_poisson_rejects_non_integer_dv(self) -> None:
        """Poisson should reject non-integer DV values."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        y = rng.uniform(0, 5, n)  # continuous, non-integer
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="poisson")

        with pytest.raises(ValueError, match="integer"):
            run_count_model(data, spec)

    def test_negbin_rejects_non_integer_dv(self) -> None:
        """NegativeBinomial should reject non-integer DV values."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        y = rng.uniform(0, 5, n)
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="negbin")

        with pytest.raises(ValueError, match="integer"):
            run_count_model(data, spec)

    def test_poisson_all_zeros_dv_raises(self) -> None:
        """Poisson should raise on all-zero DV (boundary / deviance issue)."""
        n = 50
        rng = np.random.default_rng(99)
        x1 = rng.normal(0, 1, n)
        y = np.zeros(n, dtype=int)
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="poisson")

        # All-zero DV is a boundary case — statsmodels GLM may fail with
        # "first guess on the deviance function returned a nan".  Our engine
        # should raise a ValueError rather than silently returning garbage.
        with pytest.raises(ValueError):
            run_count_model(data, spec)


# =========================================================================
# Tests: Variable labels
# =========================================================================

class TestVariableLabels:
    """Variable labels preserved for count models."""

    def test_poisson_variable_labels(self) -> None:
        """Variable labels should be present and include Intercept."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, labels = run_count_model(data, spec)

        assert len(labels) > 0
        assert "Intercept" in labels
        assert labels["Intercept"] == "Intercept"

    def test_negbin_variable_labels(self) -> None:
        """Variable labels should be present for NegBin."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, labels = run_count_model(data, spec)

        assert len(labels) > 0
        assert "Intercept" in labels

    def test_poisson_variable_labels_in_result(self) -> None:
        """Variable labels from run_count_model should appear in ModelResult."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, labels = run_count_model(data, spec)
        result = extract_count_model(fitted, variable_labels=labels)

        assert len(result.variable_labels) > 0
        assert "Intercept" in result.variable_labels


# =========================================================================
# Tests: to_dataframe with count models
# =========================================================================

class TestDataFrameCount:
    """to_dataframe() for Poisson and NegBin results."""

    def test_poisson_dataframe_z_value_column(self) -> None:
        """Poisson DataFrame should use 'z值' column header."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        df = result.to_dataframe()
        assert "z值" in df.columns, f"Expected 'z值', got {list(df.columns)}"
        assert "t值" not in df.columns

    def test_poisson_dataframe_all_finite(self) -> None:
        """All coefficient values should be finite."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        df = result.to_dataframe()
        for col in ["系数", "标准误", "z值"]:
            assert all(np.isfinite(df[col])), f"Non-finite values in {col}"

    def test_negbin_dataframe_z_value_column(self) -> None:
        """NegBin DataFrame should use 'z值' column header."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        df = result.to_dataframe()
        assert "z值" in df.columns


# =========================================================================
# Tests: to_summary_dict with count models
# =========================================================================

class TestSummaryDictCount:
    """to_summary_dict() for count model results."""

    def test_poisson_summary_dict(self) -> None:
        """Summary dict should contain count-model-specific fields."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        d = result.to_summary_dict()
        assert d["model_type"] == "poisson"
        assert d["pseudo_r_squared"] is not None
        assert d["r_squared"] is None
        assert d["rmse"] is None
        assert d["f_statistic"] is None
        assert d["f_pvalue"] is None

    def test_negbin_summary_dict(self) -> None:
        """Summary dict for NegBin should have model_type='negbin'."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        d = result.to_summary_dict()
        assert d["model_type"] == "negbin"
        assert d["pseudo_r_squared"] is not None


# =========================================================================
# Tests: summary() for count models
# =========================================================================

class TestSummaryCount:
    """summary() method for count model results."""

    def test_poisson_summary_contains_count_diagnostics(self) -> None:
        """Poisson summary should show MLE diagnostics."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        text = result.summary()
        assert "Poisson Regression Results" in text
        assert "Pseudo R-squared" in text
        assert "p>|z|" in text
        assert "p>|t|" not in text

    def test_negbin_summary_contains_method_name(self) -> None:
        """NegBin summary should show method name."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        text = result.summary()
        assert "NegativeBinomial Regression Results" in text


# =========================================================================
# Tests: is_count_model property
# =========================================================================

class TestIsCountModel:
    """is_count_model property on ModelResult."""

    def test_poisson_is_count_model(self) -> None:
        """Poisson ModelResult should be a count model."""
        result = ModelResult(
            model_type="poisson",
            coefficients=[],
            n_obs=10,
            n_params=2,
            df_resid=8,
        )
        assert result.is_count_model is True
        assert result.is_mle_model is True
        assert result.is_binary_choice is False

    def test_negbin_is_count_model(self) -> None:
        """NegBin ModelResult should be a count model."""
        result = ModelResult(
            model_type="negbin",
            coefficients=[],
            n_obs=10,
            n_params=2,
            df_resid=8,
        )
        assert result.is_count_model is True
        assert result.is_mle_model is True
        assert result.is_binary_choice is False


# =========================================================================
# Tests: Multiple predictors
# =========================================================================

class TestCountMultipleVars:
    """Count models with multiple independent variables."""

    def test_poisson_five_predictors(self) -> None:
        """Fit Poisson with 5 predictors."""
        rng = np.random.default_rng(99)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)
        x4 = rng.normal(0, 1, n)
        x5 = rng.normal(0, 1, n)
        lam = np.exp(0.8 + 0.2 * x1 - 0.1 * x2 + 0.15 * x3 + 0.05 * x4 - 0.08 * x5)
        y = rng.poisson(lam)

        df = pd.DataFrame({
            "y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
        })
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2", "x3", "x4", "x5"],
            model_type="poisson",
        )
        fitted, _ = run_count_model(df, spec)
        result = extract_count_model(fitted)

        assert len(result.coefficients) == 6  # Intercept + 5 predictors
        assert result.n_obs == n

    def test_poisson_with_control_vars(self) -> None:
        """Poisson with independent + control variables."""
        data = make_poisson_data(seed=42)
        rng = np.random.default_rng(77)
        data["x3"] = rng.normal(0, 1, len(data))

        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            control_vars=["x3"],
            model_type="poisson",
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        coef_names = {c.name for c in result.coefficients}
        assert "x3" in coef_names
        assert len(result.coefficients) == 4  # Intercept + x1 + x2 + x3


# =========================================================================
# Tests: Significance stars
# =========================================================================

class TestSignificanceStarsCount:
    """Significance stars work with count model p-values."""

    def test_stars_on_count_result(self) -> None:
        """All coefficient rows should have significance stars."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        for c in result.coefficients:
            assert c.significance in ("***", "**", "*", ""), (
                f"Unexpected significance value '{c.significance}' for {c.name}"
            )

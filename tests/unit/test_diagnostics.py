"""Tests for model diagnostics in src/modeling/diagnostics.py.

Covers: vif(), residual_tests(), influence_stats(), model_summary().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.modeling.diagnostics import (
    influence_stats,
    model_summary,
    residual_tests,
    vif,
)
from src.modeling.specification import ModelSpec
from src.results.table import CoefficientRow, ModelResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vif_data():
    """Data with one pair of correlated predictors for VIF testing."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = x1 * 0.8 + rng.normal(0, 0.3, n)  # correlated with x1
    x3 = rng.normal(0, 1, n)  # independent
    y = 2 + 0.5 * x1 + 1.0 * x2 + 0.3 * x3 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture
def fitted_ols():
    """A fitted statsmodels OLS model for influence_stats testing."""
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2 + 0.5 * x1 + 1.0 * x2 + rng.normal(0, 0.5, n)
    X = sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2}))
    return sm.OLS(y, X).fit()


@pytest.fixture
def model_result():
    """A fully populated ModelResult for model_summary testing."""
    coefs = [
        CoefficientRow(
            name="Intercept", coef=2.0, se=0.1, t_stat=20.0, pvalue=0.0001,
            ci_lower=1.8, ci_upper=2.2,
        ),
        CoefficientRow(
            name="x1", coef=0.5, se=0.05, t_stat=10.0, pvalue=0.001,
            ci_lower=0.4, ci_upper=0.6,
        ),
    ]
    return ModelResult(
        model_type="OLS",
        coefficients=coefs,
        n_obs=100,
        n_params=3,
        df_resid=97,
        r_squared=0.85,
        adj_r_squared=0.84,
        f_statistic=(150.0, 1e-20),
        log_likelihood=-120.5,
        aic=247.0,
        bic=255.0,
        rmse=0.5,
        dep_var="y",
        specification="y ~ x1 + x2",
    )


# ---------------------------------------------------------------------------
# VIF tests
# ---------------------------------------------------------------------------


class TestVIF:
    def test_vif_basic(self, vif_data):
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        result = vif(vif_data, spec)
        assert isinstance(result, pd.DataFrame)
        for col in ("variable", "vif", "vif_sqrt", "diagnosis"):
            assert col in result.columns

    def test_vif_sorted_descending(self, vif_data):
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        result = vif(vif_data, spec)
        vif_vals = result["vif"].tolist()
        assert vif_vals == sorted(vif_vals, reverse=True)

    def test_vif_diagnosis_labels(self, vif_data):
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        result = vif(vif_data, spec)
        assert set(result["diagnosis"].unique()).issubset({"High", "Moderate", "Low"})

    def test_vif_correlated_has_higher_vif(self, vif_data):
        """x1 and x2 are correlated -- they should have higher VIF than x3."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        result = vif(vif_data, spec)
        # x1 and x2 should be near top (highest VIF), x3 should be lower
        vif_x1 = result.loc[result["variable"] == "x1", "vif"].values[0]
        vif_x2 = result.loc[result["variable"] == "x2", "vif"].values[0]
        vif_x3 = result.loc[result["variable"] == "x3", "vif"].values[0]
        assert vif_x1 > vif_x3, f"Expected VIF(x1)={vif_x1} > VIF(x3)={vif_x3}"
        assert vif_x2 > vif_x3, f"Expected VIF(x2)={vif_x2} > VIF(x3)={vif_x3}"

    def test_vif_with_control_vars(self, vif_data):
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], control_vars=["x3"])
        result = vif(vif_data, spec)
        # Includes Intercept + x1 + x2 + x3 = at least 4
        assert len(result) >= 4

    def test_vif_with_categorical(self, vif_data):
        df = vif_data.copy()
        df["cat"] = np.random.choice(["A", "B", "C"], size=len(df))
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "cat"])
        result = vif(df, spec)
        assert isinstance(result, pd.DataFrame)
        # Intercept + x1 + cat dummies
        assert len(result) >= 3

    def test_vif_no_patsy(self, vif_data):
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
        result = vif(vif_data, spec, use_patsy=False)
        assert isinstance(result, pd.DataFrame)
        assert "variable" in result.columns

    def test_vif_empty_data_raises(self):
        df = pd.DataFrame({"y": pd.Series([], dtype=float), "x1": [], "x2": []})
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        with pytest.raises(ValueError):
            vif(df, spec)

    def test_vif_single_predictor(self):
        """Single predictor should still work with patsy (adds Intercept)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "y": rng.normal(0, 1, 30),
            "x1": rng.normal(0, 1, 30),
        })
        spec = ModelSpec(dep_var="y", indep_vars=["x1"])
        result = vif(df, spec)
        # Intercept + x1 = 2 rows
        assert len(result) >= 2
        assert "Intercept" in result["variable"].values


# ---------------------------------------------------------------------------
# Residual tests
# ---------------------------------------------------------------------------


class TestResidualTests:
    def test_normal_random_residuals(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        result = residual_tests(residuals)

        for key in ("shapiro_stat", "shapiro_pvalue", "shapiro_normal",
                     "dw_stat", "dw_autocorrelation"):
            assert key in result

        assert isinstance(result["shapiro_stat"], float)

    def test_dw_near_2_for_independent(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        result = residual_tests(residuals)
        assert 1.0 < result["dw_stat"] < 3.0

    def test_constant_residuals(self):
        residuals = np.ones(50)
        result = residual_tests(residuals)
        assert "dw_stat" in result
        # Constant data: scipy.shapiro may return p > 0.05 (warns zero range),
        # or may fail normality. Both are valid outcomes for edge-case data.
        assert result["shapiro_normal"] in ("Yes", "No")

    def test_positive_autocorrelation(self):
        """AR(1) with rho=0.9 should produce positive autocorrelation detection."""
        rng = np.random.default_rng(42)
        n = 100
        eps = rng.normal(0, 1, n)
        residuals = np.zeros(n)
        residuals[0] = eps[0]
        for i in range(1, n):
            residuals[i] = 0.9 * residuals[i - 1] + eps[i]
        result = residual_tests(residuals)
        assert "Positive" in result["dw_autocorrelation"]

    def test_mild_positive_autocorrelation(self):
        """AR(1) with rho=0.3 should produce mild positive detection."""
        rng = np.random.default_rng(42)
        n = 100
        eps = rng.normal(0, 1, n)
        residuals = np.zeros(n)
        residuals[0] = eps[0]
        for i in range(1, n):
            residuals[i] = 0.3 * residuals[i - 1] + eps[i]
        result = residual_tests(residuals)
        assert "Positive (mild)" in result["dw_autocorrelation"]

    def test_negative_autocorrelation(self):
        """AR(1) with rho=-0.9 produces strong negative autocorrelation."""
        rng = np.random.default_rng(42)
        n = 100
        eps = rng.normal(0, 1, n)
        residuals = np.zeros(n)
        residuals[0] = eps[0]
        for i in range(1, n):
            residuals[i] = -0.9 * residuals[i - 1] + eps[i]
        result = residual_tests(residuals)
        assert "Negative" in result["dw_autocorrelation"]

    def test_insufficient_data_2_points(self):
        residuals = np.array([1.0, 2.0])
        result = residual_tests(residuals)
        assert result["shapiro_normal"] == "Insufficient data"
        # DW should be computable with 2 points; it gives a valid number
        assert isinstance(result["dw_stat"], float)

    def test_insufficient_data_1_point(self):
        residuals = np.array([1.0])
        result = residual_tests(residuals)
        assert result["shapiro_normal"] == "Insufficient data"
        assert result["dw_autocorrelation"] == "Insufficient data"

    def test_empty_residuals(self):
        residuals = np.array([], dtype=float)
        result = residual_tests(residuals)
        assert result["shapiro_normal"] == "Insufficient data"
        assert result["dw_autocorrelation"] == "Insufficient data"


# ---------------------------------------------------------------------------
# Influence statistics
# ---------------------------------------------------------------------------


class TestInfluenceStats:
    def test_basic(self, fitted_ols):
        result = influence_stats(fitted_ols)
        assert isinstance(result, pd.DataFrame)
        for col in ("cooks_d", "leverage", "observation"):
            assert col in result.columns
        assert len(result) == 100

    def test_cooks_d_non_negative(self, fitted_ols):
        result = influence_stats(fitted_ols)
        assert (result["cooks_d"] >= 0).all()

    def test_leverage_between_0_and_1(self, fitted_ols):
        result = influence_stats(fitted_ols)
        assert (result["leverage"] >= 0).all()
        assert (result["leverage"] <= 1).all()

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="get_influence"):
            influence_stats("not_a_model")

    def test_none_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            influence_stats(None)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------


class TestModelSummary:
    def test_basic_fields(self, model_result):
        summary = model_summary(model_result)
        assert isinstance(summary, dict)
        assert summary["model_type"] == "OLS"
        assert summary["dep_var"] == "y"
        assert summary["n_obs"] == 100
        assert summary["r_squared"] == 0.85
        assert summary["n_params"] == 3

    def test_coefficients_list(self, model_result):
        summary = model_summary(model_result)
        assert "coefficients" in summary
        coefs = summary["coefficients"]
        assert len(coefs) == 2
        assert coefs[0]["变量"] == "Intercept"

    def test_without_f_statistic(self):
        """ModelResult with f_statistic=None should not include F fields."""
        coefs = [
            CoefficientRow(
                name="Intercept", coef=1.0, se=0.2, t_stat=5.0, pvalue=0.01,
                ci_lower=0.6, ci_upper=1.4,
            ),
        ]
        result = ModelResult(
            model_type="OLS", coefficients=coefs,
            n_obs=10, n_params=1, df_resid=9,
            r_squared=0.5, adj_r_squared=0.4,
            f_statistic=None, log_likelihood=-10.0,
            aic=20.0, bic=22.0, rmse=1.0,
            dep_var="y", specification="y ~ 1",
        )
        summary = model_summary(result)
        assert "f_statistic" not in summary

    def test_without_log_likelihood(self):
        """ModelResult with log_likelihood=None should not include it."""
        coefs = [
            CoefficientRow(
                name="Intercept", coef=1.0, se=0.2, t_stat=5.0, pvalue=0.01,
                ci_lower=0.6, ci_upper=1.4,
            ),
        ]
        result = ModelResult(
            model_type="OLS", coefficients=coefs,
            n_obs=10, n_params=1, df_resid=9,
            r_squared=0.5, adj_r_squared=0.4,
            f_statistic=(10.0, 0.01), log_likelihood=None,
            aic=20.0, bic=22.0, rmse=1.0,
            dep_var="y", specification="y ~ 1",
        )
        summary = model_summary(result)
        assert "log_likelihood" not in summary

    def test_includes_specification(self, model_result):
        summary = model_summary(model_result)
        assert summary["specification"] == "y ~ x1 + x2"

    def test_includes_method(self, model_result):
        summary = model_summary(model_result)
        assert summary["method"] == "OLS"

    def test_includes_rmse(self, model_result):
        summary = model_summary(model_result)
        assert summary["rmse"] == 0.5

    def test_single_coefficient_works(self):
        """ModelResult with a single coefficient should produce summary."""
        result = ModelResult(
            model_type="WLS", coefficients=[
                CoefficientRow(
                    name="Intercept", coef=3.0, se=0.5, t_stat=6.0, pvalue=0.001,
                    ci_lower=2.0, ci_upper=4.0,
                ),
            ],
            n_obs=50, n_params=2, df_resid=48,
            r_squared=0.3, adj_r_squared=0.28,
            f_statistic=(5.0, 0.03), log_likelihood=None,
            aic=100.0, bic=105.0, rmse=2.0,
            dep_var="z", specification="z ~ w",
        )
        summary = model_summary(result)
        assert summary["dep_var"] == "z"
        assert len(summary["coefficients"]) == 1

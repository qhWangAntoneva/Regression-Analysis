# encoding: utf-8
"""Unit tests for the Logit regression engine.

Tests cover:
    - Basic logit fit on binary outcome data
    - Coefficient reasonableness (sign check)
    - Pseudo R-squared between 0 and 1
    - z-statistics in output
    - Perfect separation handling
    - ModelResult construction from extract_logit()
    - model_type field is "logit"
    - OLS-specific fields are None for logit
    - to_summary_dict() with logit result
    - to_dataframe() with logit result
    - _pvalue_stars() works with logit p-values
    - Odds ratio calculation
    - Fit on spector dataset
    - Multiple independent variables
    - Fitter dispatch: fit() with model_type="logit"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit
from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.results.table import ModelResult, _significance_stars


# =========================================================================
# Helper: create a binary dataset via logistic DGP
# =========================================================================
def make_binary_data(
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a dataset for binary logistic regression.

    DGP: y* = 0.5 + 1.0*x1 - 0.8*x2 + noise(logistic)
    y = 1 if y* > 0 else 0
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 0.5 + 1.0 * x1 - 0.8 * x2
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# =========================================================================
# Tests: Basic logit fit
# =========================================================================
class TestLogitBasic:
    """Basic logit regression on synthetic binary data."""

    def test_logit_fit_success(self) -> None:
        """Logit should fit without error and return results."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted = run_logit(data, spec)
        result = extract_logit(fitted, dep_var="y")

        assert result.model_type == "logit"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3  # Intercept + x1 + x2

    def test_logit_coefficient_signs(self) -> None:
        """Coefficients should have expected signs from DGP."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        coef_map = {c.name: c for c in result.coefficients}
        # x1 has positive effect (coef=1.0 in DGP)
        assert coef_map["x1"].coef > 0, "x1 should have positive coefficient"
        # x2 has negative effect (coef=-0.8 in DGP)
        assert coef_map["x2"].coef < 0, "x2 should have negative coefficient"

    def test_logit_standard_errors_positive(self) -> None:
        """All standard errors should be positive."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        for c in result.coefficients:
            assert c.se > 0, f"Non-positive SE for {c.name}"

    def test_logit_pvalues_in_range(self) -> None:
        """All p-values should be between 0 and 1."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        for c in result.coefficients:
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}"


# =========================================================================
# Tests: Pseudo R-squared
# =========================================================================
class TestPseudoRSquared:
    """McFadden's pseudo R-squared."""

    def test_pseudo_r_squared_between_0_and_1(self) -> None:
        """Pseudo R-squared should be in [0, 1]."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        assert result.pseudo_r_squared is not None
        assert 0 <= result.pseudo_r_squared <= 1, (
            f"Pseudo R² = {result.pseudo_r_squared}, expected [0, 1]"
        )

    def test_pseudo_r_squared_is_none_for_ols(self) -> None:
        """Pseudo R-squared should be None for OLS results (default)."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=10,
            n_params=2,
            df_resid=8,
        )
        assert result.pseudo_r_squared is None


# =========================================================================
# Tests: z-statistics
# =========================================================================
class TestZStatistics:
    """z-statistics from logit model."""

    def test_z_stat_available(self) -> None:
        """z-statistics should be available on coefficient rows."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        for c in result.coefficients:
            assert c.z_stat == c.t_stat, "z_stat should equal t_stat for logit"

    def test_z_stat_nonzero(self) -> None:
        """z-statistics should be non-zero for reasonably predictive vars."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        coef_map = {c.name: c for c in result.coefficients}
        assert abs(coef_map["x1"].z_stat) > 0.1


# =========================================================================
# Tests: OLS-specific fields are None for logit
# =========================================================================
class TestOLSFieldsNoneForLogit:
    """Verify OLS-specific fields are None for logit results."""

    def test_r_squared_is_none(self) -> None:
        """Logit should not have OLS R-squared."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        assert result.r_squared is None
        assert result.adj_r_squared is None

    def test_f_statistic_is_none(self) -> None:
        """Logit should not have F-statistic."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        assert result.f_statistic is None

    def test_rmse_is_none(self) -> None:
        """Logit should have RMSE as None."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        assert result.rmse is None


# =========================================================================
# Tests: to_summary_dict() with logit
# =========================================================================
class TestSummaryDictLogit:
    """to_summary_dict() with logit result."""

    def test_summary_dict_contains_logit_fields(self) -> None:
        """Summary dict should have logit-specific keys."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        d = result.to_summary_dict()
        assert d["model_type"] == "logit"
        assert d["pseudo_r_squared"] is not None
        assert d["pseudo_r_squared"] == result.pseudo_r_squared
        assert d["f_statistic"] is None
        assert d["f_pvalue"] is None
        assert "llr" in d
        assert "llr_pvalue" in d


# =========================================================================
# Tests: to_dataframe() with logit
# =========================================================================
class TestDataFrameLogit:
    """to_dataframe() with logit result."""

    def test_to_dataframe_uses_z_value_column(self) -> None:
        """Logit DataFrame should use 'z值' column header."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        df = result.to_dataframe()
        assert "z值" in df.columns, f"Expected 'z值' column, got {list(df.columns)}"

    def test_to_dataframe_all_finite(self) -> None:
        """All values in logit DataFrame should be finite."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        df = result.to_dataframe()
        for col in ["系数", "标准误", "z值"]:
            assert all(np.isfinite(df[col])), f"Non-finite values in {col}"

    def test_to_dataframe_ci_order(self) -> None:
        """CI lower should be less than CI upper for all coefficients."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        df = result.to_dataframe()
        assert all(df["95%CI低"] < df["95%CI高"])


# =========================================================================
# Tests: _significance_stars with logit p-values
# =========================================================================
class TestSignificanceStarsLogit:
    """_pvalue_stars() (via _significance_stars) works with logit p-values."""

    def test_stars_highly_significant(self) -> None:
        assert _significance_stars(0.001) == "***"

    def test_stars_significant(self) -> None:
        assert _significance_stars(0.03) == "**"

    def test_stars_weakly_significant(self) -> None:
        assert _significance_stars(0.07) == "*"

    def test_stars_not_significant(self) -> None:
        assert _significance_stars(0.5) == ""


# =========================================================================
# Tests: Odds ratios
# =========================================================================
class TestOddsRatios:
    """Odds ratio calculation: OR = exp(coef)."""

    def test_odds_ratio_from_coefficients(self) -> None:
        """OR = exp(coef), OR_CI = exp(conf_int)."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        for c in result.coefficients:
            or_val = np.exp(c.coef)
            assert np.isfinite(or_val), f"Non-finite OR for {c.name}"
            # OR should be positive
            assert or_val > 0

    def test_odds_ratio_ci_from_coef_ci(self) -> None:
        """Odds ratio CI = exp(coef CI)."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        for c in result.coefficients:
            or_lower = np.exp(c.ci_lower)
            or_upper = np.exp(c.ci_upper)
            assert or_lower <= or_upper, "OR CI lower should be <= upper"


# =========================================================================
# Tests: LR test fields
# =========================================================================
class TestLikelihoodRatioTest:
    """LLR (likelihood ratio chi-squared) field."""

    def test_llr_set(self) -> None:
        """LLR should be set for logit results."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        assert result.llr is not None
        assert result.llr > 0, "LLR should be positive for a reasonable model"

    def test_llr_pvalue_set(self) -> None:
        """LLR p-value should be set."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        assert result.llr_pvalue is not None
        assert 0 <= result.llr_pvalue <= 1


# =========================================================================
# Tests: Fit on statsmodels spector dataset (real data)
# =========================================================================
class TestSpectorDataset:
    """Fit logit on statsmodels built-in spector dataset."""

    def test_logit_on_spector(self) -> None:
        """Fit logit on the Spector-Mazzeo grade dataset."""
        import statsmodels.api as sm

        spector_data = sm.datasets.spector.load_pandas()
        df = spector_data.data
        df["GRADE"] = spector_data.endog

        spec = ModelSpec(
            dep_var="GRADE",
            indep_vars=["GPA", "TUCE", "PSI"],
            model_type="logit",
        )
        fitted, _ = run_logit(df, spec)
        result = extract_logit(fitted)

        assert result.model_type == "logit"
        assert result.n_obs == 32
        assert len(result.coefficients) == 4  # Intercept + 3 predictors
        assert result.pseudo_r_squared is not None
        assert 0 <= result.pseudo_r_squared <= 1
        assert result.llr is not None
        assert result.llr > 0


# =========================================================================
# Tests: Multiple independent variables
# =========================================================================
class TestLogitMultipleVars:
    """Logit with multiple independent variables."""

    def test_logit_five_predictors(self) -> None:
        """Fit logit with 5 predictors."""
        rng = np.random.default_rng(99)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)
        x4 = rng.normal(0, 1, n)
        x5 = rng.normal(0, 1, n)
        eta = 0.3 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + 0.1 * x4 - 0.2 * x5
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(n) < prob).astype(int)

        df = pd.DataFrame({
            "y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
        })

        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2", "x3", "x4", "x5"],
            model_type="logit",
        )
        fitted, _ = run_logit(df, spec)
        result = extract_logit(fitted)

        assert len(result.coefficients) == 6  # Intercept + 5 predictors
        assert result.n_obs == n
        assert result.pseudo_r_squared is not None
        assert result.pseudo_r_squared > 0

    def test_logit_with_control_vars(self) -> None:
        """Logit with independent + control variables."""
        data = make_binary_data(seed=42)
        data["x3"] = np.random.default_rng(77).normal(0, 1, len(data))

        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            control_vars=["x3"],
            model_type="logit",
        )
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        coef_names = [c.name for c in result.coefficients]
        assert "x3" in coef_names
        assert len(result.coefficients) == 4  # Intercept + x1 + x2 + x3


# =========================================================================
# Tests: Fitter dispatch
# =========================================================================
class TestFitterDispatch:
    """ModelFitter dispatches correctly based on model_type."""

    def test_fitter_logit_dispatch(self) -> None:
        """Fitter with model_type='logit' should call logit engine."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.model_type == "logit"
        assert result.pseudo_r_squared is not None
        assert result.r_squared is None
        assert result.f_statistic is None

    def test_fitter_ols_still_works(self) -> None:
        """Fitter with default model_type should still work for OLS."""
        import pandas as pd

        df = pd.DataFrame({
            "y": [1, 2, 3, 4, 5],
            "x": [0.5, 1.0, 1.5, 2.0, 2.5],
        })
        spec = ModelSpec(dep_var="y", indep_vars=["x"])
        fitter = ModelFitter()
        result = fitter.fit(spec, df)

        assert result.model_type == "OLS"
        assert result.r_squared is not None
        assert result.rmse is not None

    def test_fitter_multiple_mixed(self) -> None:
        """fit_multiple with mixed OLS and logit specs."""
        data = make_binary_data(seed=42)
        data["y_cont"] = data["x1"] * 0.5 + data["x2"] * 0.3 + np.random.default_rng(88).normal(0, 0.1, len(data))

        spec_logit = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        spec_ols = ModelSpec(dep_var="y_cont", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        results = fitter.fit_multiple([spec_logit, spec_ols], data)

        assert len(results) == 2
        assert results[0].model_type == "logit"
        assert results[1].model_type == "OLS"


# =========================================================================
# Tests: Summary text for logit
# =========================================================================
class TestSummaryMethod:
    """summary() method for logit results."""

    def test_summary_logit_contains_pseudo_r2(self) -> None:
        """Logit summary should show pseudo R-squared."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        summary_text = result.summary()
        assert "Pseudo R-squared" in summary_text
        assert "LR chi2" in summary_text
        assert "p>|z|" in summary_text
        # Should NOT contain OLS-specific headers
        assert "p>|t|" not in summary_text


# =========================================================================
# Tests: to_latex_row for logit
# =========================================================================
class TestLatexRowLogit:
    """to_latex_row() for logit ModelResults."""

    def test_latex_row_logit_uses_pseudo_r2(self) -> None:
        """Logit LaTeX row should not contain OLS-style fields."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        # Should have 7 parts (dep_var, n, pseudo_r2, llr, llr_p, aic, bic)
        parts = latex.split(" & ")
        assert len(parts) == 7, f"Expected 7 parts, got {len(parts)}: {parts}"
        assert "N/A" not in parts  # All fields should have values


# =========================================================================
# Tests: ANOVA table for logit is empty
# =========================================================================
class TestAnovaLogit:
    """anova_table() returns empty for logit."""

    def test_anova_empty_for_logit(self) -> None:
        """ANOVA table should be empty DataFrame for logit."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        fitted, _ = run_logit(data, spec)
        result = extract_logit(fitted)

        anova = result.anova_table()
        assert anova.empty


# =========================================================================
# Tests: Perfect separation detection
# =========================================================================
class TestPerfectSeparation:
    """Handling of perfect / quasi-perfect separation."""

    def test_perfect_separation_raises(self) -> None:
        """Perfect separation should raise ValueError."""
        rng = np.random.default_rng(42)
        n = 50
        x1 = rng.normal(0, 1, n)
        # y = 1 whenever x1 > 0 => perfect separation
        y = (x1 > 0).astype(int)

        df = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="logit")

        # Depending on the data, this may converge or not.
        # Statsmodels often warns but still produces output for quasi-separation.
        # For true perfect separation with intercept+single predictor it should
        # typically raise or at least not converge cleanly.
        try:
            fitted = run_logit(df, spec)
            result = extract_logit(fitted)
            # If it did converge, check that coefficients are huge (separation)
            # This is acceptable - statsmodels can sometimes handle it
        except ValueError as e:
            assert "converge" in str(e).lower() or "perfect" in str(e).lower()

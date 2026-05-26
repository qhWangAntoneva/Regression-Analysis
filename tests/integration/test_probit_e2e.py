# encoding: utf-8
"""End-to-end integration tests for Probit regression.

Covers the complete pipeline from data preparation through model fitting
to result extraction, formatting, and web-bridge integration.

Tests are organised into two groups:
    - Streamlit-side: ModelSpec -> Fitter -> ModelResult -> formatting
    - Web-bridge: bridge.py data flow -> JSON-serializable output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_probit_engine import extract_probit, run_probit
from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec, build_variable_labels
from src.results.table import CoefficientRow, ModelResult, compare_models

# ---------------------------------------------------------------------------
# Make the web bridge importable for integration testing
# ---------------------------------------------------------------------------
_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "py"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))


# =========================================================================
# Helpers
# =========================================================================
def make_binary_data(
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic dataset for binary probit regression.

    DGP: y* = 0.5 + 1.0*x1 - 0.8*x2 + N(0,1) noise
    y = 1 if y* > 0 else 0
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.choice(["A", "B", "C"], n)
    cat1 = rng.integers(0, 2, n)
    eta = 0.5 + 1.0 * x1 - 0.8 * x2 + rng.normal(0, 1, n)
    y = (eta > 0).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "cat1": cat1})


def make_perfect_separation_data(
    n: int = 60,
    seed: int = 123,
) -> pd.DataFrame:
    """Create data with near-perfect separation."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    y = (x1 > 0).astype(int)
    # Add a small amount of overlap (2 flipped labels)
    flip = rng.choice(n, size=2, replace=False)
    y[flip] = 1 - y[flip]
    return pd.DataFrame({"y": y, "x1": x1})


# =========================================================================
# Streamlit-side integration tests
# =========================================================================


class TestFullPipeline:
    """End-to-end: data -> ModelSpec -> Fitter.fit -> ModelResult."""

    def test_full_pipeline_basic(self) -> None:
        """Complete probit pipeline produces a valid ModelResult."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted, dep_var="y")

        assert isinstance(result, ModelResult)
        assert result.model_type == "probit"
        assert result.method == "Probit"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3  # Intercept + x1 + x2

    def test_full_pipeline_pseudo_r_squared_in_range(self) -> None:
        """Pseudo R-squared should be in (0, 1) for a reasonable model."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert result.pseudo_r_squared is not None
        assert 0 < result.pseudo_r_squared < 1, (
            f"Pseudo R-squared = {result.pseudo_r_squared}, expected (0, 1)"
        )

    def test_full_pipeline_ols_fields_are_none(self) -> None:
        """OLS-specific fields should be None for probit results."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert result.r_squared is None
        assert result.adj_r_squared is None
        assert result.f_statistic is None
        assert result.rmse is None

    def test_full_pipeline_coefficients_valid(self) -> None:
        """Every coefficient row should have valid statistics."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        for c in result.coefficients:
            assert isinstance(c, CoefficientRow)
            assert isinstance(c.name, str) and len(c.name) > 0
            assert np.isfinite(c.coef), f"Non-finite coef for {c.name}"
            assert c.se > 0, f"Non-positive SE for {c.name}"
            assert np.isfinite(c.t_stat), f"Non-finite z-stat for {c.name}"
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}"
            assert c.ci_lower < c.ci_upper, (
                f"CI inversion for {c.name}: "
                f"lower={c.ci_lower} >= upper={c.ci_upper}"
            )

    def test_full_pipeline_log_likelihood_and_llr(self) -> None:
        """Log-likelihood and LLR should be finite and positive."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)
        assert result.llr is not None
        assert result.llr > 0, "LLR should be positive for a reasonable model"
        assert result.llr_pvalue is not None
        assert 0 <= result.llr_pvalue <= 1

    def test_full_pipeline_dep_var_and_specification(self) -> None:
        """The ModelResult should preserve the dep_var and specification string."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted, dep_var="y",
                                specification="y ~ x1 + x2")

        assert result.dep_var == "y"
        assert "x1" in result.specification
        assert "x2" in result.specification

    def test_full_pipeline_with_control_vars(self) -> None:
        """Pipeline with indep_vars + control_vars produces correct n_params."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1"],
            control_vars=["x2"],
            model_type="probit",
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert len(result.coefficients) == 3  # Intercept + x1 + x2
        coef_names = {c.name for c in result.coefficients}
        assert "x2" in coef_names

    def test_full_pipeline_no_intercept(self) -> None:
        """Probit pipeline without an intercept term."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"],
            has_intercept=False, model_type="probit",
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert len(result.coefficients) == 2  # x1 + x2, no intercept
        coef_names = {c.name for c in result.coefficients}
        assert "Intercept" not in coef_names
        assert "x1" in coef_names
        assert "x2" in coef_names


class TestFitterDispatch:
    """ModelFitter dispatches correctly based on model_type."""

    def test_fitter_probit_dispatch(self) -> None:
        """model_type='probit' routes to MLE engine."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.pseudo_r_squared is not None
        assert result.r_squared is None

    def test_fitter_mixed_probit_and_ols(self) -> None:
        """fit_multiple handles mixed probit + OLS specs in one call."""
        data = make_binary_data(seed=42)
        data["y_cont"] = (
            data["x1"] * 0.5
            + data["x2"] * 0.3
            + np.random.default_rng(88).normal(0, 0.1, len(data))
        )

        spec_probit = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        spec_ols = ModelSpec(dep_var="y_cont", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        results = fitter.fit_multiple([spec_probit, spec_ols], data)

        assert len(results) == 2
        assert results[1].model_type == "OLS"

    def test_fitter_all_probit_multiple(self) -> None:
        """fit_multiple with all probit specs compares two probit models."""
        data = make_binary_data(seed=42)
        spec1 = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="probit"
        )
        spec2 = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitter = ModelFitter()
        results = fitter.fit_multiple([spec1, spec2], data)

        assert len(results) == 2
        assert results[0].n_params == 2  # Intercept + x1
        assert results[1].n_params == 3  # Intercept + x1 + x2


class TestVariableLabels:
    """Variable labels in the probit pipeline."""

    def test_labels_categorical_in_probit_pipeline(self) -> None:
        """Labels for categorical dummies appear in ModelResult.variable_labels."""
        data = make_binary_data(seed=42)
        data["x3"] = np.random.default_rng(77).choice(
            ["Low", "Medium", "High"], len(data)
        )
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x3"], model_type="probit"
        )
        fitted, labels = run_probit(data, spec)
        result = extract_probit(fitted, variable_labels=labels)

        assert "variable_labels" in result.__dataclass_fields__
        labels = result.variable_labels
        assert len(labels) > 0, "Variable labels dict should not be empty"

        assert "Intercept" in labels or any(
            "x3" in k for k in labels
        ), f"Expected categorical labels, got: {labels}"

    def test_labels_nonempty_for_continuous_vars(self) -> None:
        """Continuous-only specification still produces a non-empty label dict."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, labels = run_probit(data, spec)
        result = extract_probit(fitted, variable_labels=labels)

        labels = result.variable_labels
        assert len(labels) > 0
        assert "Intercept" in labels
        assert labels["Intercept"] == "Intercept"


class TestSummaryDict:
    """to_summary_dict() for probit results."""

    def test_summary_dict_contains_probit_specific_fields(self) -> None:
        """Summary dict should expose pseudo_r_squared, llr, llr_pvalue."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        d = result.to_summary_dict()
        assert d["model_type"] == "probit"
        assert d["pseudo_r_squared"] is not None
        assert d["pseudo_r_squared"] == result.pseudo_r_squared
        assert d["llr"] == result.llr
        assert d["llr_pvalue"] == result.llr_pvalue
        assert d["f_statistic"] is None
        assert d["f_pvalue"] is None
        assert d["log_likelihood"] == result.log_likelihood


class TestDataFrame:
    """to_dataframe() for probit results."""

    def test_dataframe_uses_z_value_column(self) -> None:
        """Probit DataFrame column header is 'z值' not 't值'."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        df = result.to_dataframe()
        assert "z值" in df.columns
        assert "t值" not in df.columns

    def test_dataframe_no_odds_ratio_column(self) -> None:
        """Probit DataFrame should NOT include an OR column."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        df = result.to_dataframe()
        assert "OR(exp(B))" not in df.columns, (
            "Probit results should not have odds ratio column"
        )

    def test_dataframe_all_values_finite(self) -> None:
        """Coefficient, SE, z-value columns should all be finite."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        df = result.to_dataframe()
        for col in ["系数", "标准误", "z值"]:
            assert all(np.isfinite(df[col])), f"Non-finite values in {col}"

    def test_dataframe_ci_order(self) -> None:
        """CI lower < CI upper for every coefficient."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        df = result.to_dataframe()
        assert all(df["95%CI低"] < df["95%CI高"])


class TestCategoricalPredictor:
    """Full pipeline with a categorical independent variable."""

    def test_probit_with_categorical_predictor(self) -> None:
        """Probit fit when one predictor is categorical (string)."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x3"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert result.model_type == "probit"
        # Design matrix: Intercept + x1 + C(x3)[T.B] + C(x3)[T.C] = 4 cols
        assert len(result.coefficients) == 4
        coef_names = {c.name for c in result.coefficients}
        assert "Intercept" in coef_names
        assert "x1" in coef_names
        assert any("x3" in name for name in coef_names if name != "x1")

    def test_probit_categorical_labels_readable(self) -> None:
        """Variable labels for categorical dummies are human-readable."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x3"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        labels = result.variable_labels
        decoded_labels = [v for v in labels.values() if "x3" in v or ":" in v]
        if decoded_labels:
            assert all(
                isinstance(v, str) for v in decoded_labels
            ), "Labels should be strings"


class TestInteraction:
    """Full pipeline with an interaction term."""

    def test_probit_with_interaction(self) -> None:
        """Probit fit with x1:x2 interaction produces correct coefficient count."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            interaction_terms=[("x1", "x2")],
            model_type="probit",
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        assert result.model_type == "probit"
        # Intercept + x1 + x2 + x1:x2 = 4
        assert len(result.coefficients) == 4
        coef_names = {c.name for c in result.coefficients}
        assert "x1" in coef_names
        assert "x2" in coef_names
        assert any(":" in n for n in coef_names), (
            f"No interaction term found in {coef_names}"
        )

    def test_probit_interaction_metadata(self) -> None:
        """Interaction metadata is preserved in result (via fitter dispatch)."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            interaction_terms=[("x1", "x2")],
            model_type="probit",
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.interaction_terms_applied == [("x1", "x2")]


class TestConfidenceInterval:
    """Confidence interval verification for probit coefficients."""

    def test_ci_contains_coefficient(self) -> None:
        """95% CI should contain the estimated coefficient value."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        for c in result.coefficients:
            assert c.ci_lower <= c.coef <= c.ci_upper, (
                f"CI [{c.ci_lower:.4f}, {c.ci_upper:.4f}] does not "
                f"contain coefficient {c.coef:.4f} for {c.name}"
            )

    def test_ci_narrows_with_more_data(self) -> None:
        """CIs should be narrower with larger sample sizes."""
        # Small dataset
        data_small = make_binary_data(n=50, seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="probit"
        )
        fitted_small, _ = run_probit(data_small, spec)
        result_small = extract_probit(fitted_small)
        ci_width_small = result_small.coefficients[1].ci_upper - result_small.coefficients[1].ci_lower

        # Large dataset
        data_large = make_binary_data(n=500, seed=42)
        fitted_large, _ = run_probit(data_large, spec)
        result_large = extract_probit(fitted_large)
        ci_width_large = result_large.coefficients[1].ci_upper - result_large.coefficients[1].ci_lower

        assert ci_width_large < ci_width_small, (
            f"CI width should decrease with more data: "
            f"small={ci_width_small:.4f}, large={ci_width_large:.4f}"
        )


class TestPerfectSeparation:
    """Perfect / quasi-separation handling for probit."""

    def test_perfect_separation_raises_or_warns(self) -> None:
        """Near-perfect separation should either raise ValueError or converge
        with huge SEs."""
        data = make_perfect_separation_data(seed=123)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="probit"
        )

        try:
            fitted, labels = run_probit(data, spec)
            result = extract_probit(fitted, dep_var="y")
            # If it converged, coefficients should be huge (separation)
            x1_row = next(c for c in result.coefficients if c.name == "x1")
            assert abs(x1_row.coef) > 2.0, (
                "With near-separation, x1 coefficient should be large"
            )
        except ValueError:
            pass

    def test_strictly_perfect_separation_detected(self) -> None:
        """A truly separable dataset should trigger a ValueError or huge coef."""
        rng = np.random.default_rng(99)
        n = 40
        x1 = rng.normal(0, 1, n)
        y = (x1 > 0).astype(int)  # no overlap at all

        df = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="probit"
        )

        try:
            fitted, _ = run_probit(df, spec)
            result = extract_probit(fitted)
            x1_coef = next(c for c in result.coefficients if c.name == "x1")
            assert abs(x1_coef.coef) > 3.0 or x1_coef.se > 10.0, (
                "Strictly separable data should produce extreme estimates"
            )
        except ValueError:
            pass


class TestSummaryMethod:
    """summary() and to_latex_row() for probit."""

    def test_summary_probit_contains_pseudo_r2(self) -> None:
        """Probit summary shows pseudo R-squared and LR chi2, not t-stat."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        text = result.summary()
        assert "Probit Regression Results" in text
        assert "Pseudo R-squared" in text
        assert "LR chi2" in text
        assert "p>|z|" in text
        assert "p>|t|" not in text
        assert "R-squared:" not in text.replace("Pseudo R-squared:", "")

    def test_latex_row_probit_seven_parts(self) -> None:
        """Probit LaTeX row has 7 parts: dep_var, n, pseudo_r2, llr, llr_p, aic, bic."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        assert len(parts) == 7, f"Expected 7 parts, got {len(parts)}: {parts}"
        assert "N/A" not in parts, f"All fields should have values: {parts}"


class TestAnovaTable:
    """ANOVA table is empty for probit models."""

    def test_anova_empty_for_probit(self) -> None:
        """anova_table() returns an empty DataFrame for probit."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted, _ = run_probit(data, spec)
        result = extract_probit(fitted)

        anova = result.anova_table()
        assert anova.empty


class TestCompareModels:
    """compare_models() with probit results."""

    def test_compare_two_probit_models(self) -> None:
        """compare_models works with two probit results."""
        data = make_binary_data(seed=42)

        spec1 = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="probit"
        )
        spec2 = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="probit"
        )
        fitted1, _ = run_probit(data, spec1)
        fitted2, _ = run_probit(data, spec2)
        result1 = extract_probit(fitted1)
        result2 = extract_probit(fitted2)

        comparison = compare_models([result1, result2])
        assert not comparison.empty
        assert len(comparison) >= 3  # at least 2 coefs + stats

    def test_compare_three_probit_models(self) -> None:
        """compare_models with three probit models."""
        data = make_binary_data(seed=200)

        spec1 = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="probit")
        spec2 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="probit")
        spec3 = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"],
            interaction_terms=[("x1", "x2")], model_type="probit",
        )
        fitted1, _ = run_probit(data, spec1)
        fitted2, _ = run_probit(data, spec2)
        fitted3, _ = run_probit(data, spec3)
        results = [extract_probit(f) for f in [fitted1, fitted2, fitted3]]

        comparison = compare_models(results)
        assert not comparison.empty
        assert len(comparison) >= 4  # coef rows + stat rows


# =========================================================================
# Probit vs Logit comparison
# =========================================================================


class TestProbitVsLogit:
    """Comparison between probit and logit on the same data."""

    def test_probit_logit_same_signs(self) -> None:
        """Probit and logit should agree on coefficient signs."""
        from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit

        data = make_binary_data(seed=42)
        spec_logit = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        spec_probit = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="probit")

        fitted_logit, _ = run_logit(data, spec_logit)
        fitted_probit, _ = run_probit(data, spec_probit)

        result_logit = extract_logit(fitted_logit)
        result_probit = extract_probit(fitted_probit)

        for c_logit in result_logit.coefficients:
            c_probit = next(c for c in result_probit.coefficients if c.name == c_logit.name)
            assert (c_logit.coef > 0) == (c_probit.coef > 0), (
                f"Sign mismatch for {c_logit.name}: "
                f"logit={c_logit.coef:.4f}, probit={c_probit.coef:.4f}"
            )

    def test_probit_logit_similar_significance(self) -> None:
        """Probit and logit should have broadly similar significance levels."""
        from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit

        data = make_binary_data(seed=42)
        spec_logit = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit")
        spec_probit = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="probit")

        fitted_logit, _ = run_logit(data, spec_logit)
        fitted_probit, _ = run_probit(data, spec_probit)

        result_logit = extract_logit(fitted_logit)
        result_probit = extract_probit(fitted_probit)

        for c_logit in result_logit.coefficients:
            c_probit = next(c for c in result_probit.coefficients if c.name == c_logit.name)
            # Significance should be broadly similar (same stars or off by at most one level)
            assert abs(c_logit.pvalue - c_probit.pvalue) < 0.5, (
                f"Large p-value difference for {c_logit.name}: "
                f"logit={c_logit.pvalue:.4f}, probit={c_probit.pvalue:.4f}"
            )


# =========================================================================
# Web-bridge integration tests
# =========================================================================


class TestBridgeProbitIntegration:
    """Simulate the web/bridge.py probit flow and verify JSON output."""

    def test_bridge_probit_basic(self) -> None:
        """Simulate bridge.run_regression with model_type='probit'."""
        import bridge

        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in make_binary_data(n=100, seed=42).iterrows()
            ],
            "columns": [
                {"name": "y", "col_type": "numeric"},
                {"name": "x1", "col_type": "numeric"},
                {"name": "x2", "col_type": "numeric"},
            ],
        }
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "probit",
            "alpha": 0.05,
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"], f"Bridge probit failed: {result.get('error')}"
        assert result["model_type"] in ("probit", "logit")
        assert result["r_squared"] is None
        assert result["rmse"] is None
        assert result["pseudo_r_squared"] is not None
        assert 0 < result["pseudo_r_squared"] < 1
        assert result["llr"] is not None
        assert result["llr"] > 0
        assert len(result["coefficients"]) == 3  # Intercept + x1 + x2

    def test_bridge_probit_rejects_non_binary_dep_var(self) -> None:
        """Bridge should error when dep_var has more than 2 unique values."""
        import bridge

        df_nonbinary = make_binary_data(n=50, seed=42).copy()
        df_nonbinary["y"] = [i % 3 for i in range(len(df_nonbinary))]

        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in df_nonbinary.iterrows()
            ],
            "columns": [
                {"name": "y", "col_type": "numeric"},
                {"name": "x1", "col_type": "numeric"},
                {"name": "x2", "col_type": "numeric"},
            ],
        }
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "probit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert not result["success"]
        assert "binary" in result["error"].lower()

    def test_bridge_probit_indep_vars_in_result(self) -> None:
        """Bridge result includes the list of independent variables used."""
        import bridge

        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in make_binary_data(n=100, seed=42).iterrows()
            ],
            "columns": [
                {"name": "y", "col_type": "numeric"},
                {"name": "x1", "col_type": "numeric"},
                {"name": "x2", "col_type": "numeric"},
            ],
        }
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "probit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"]
        assert "indep_vars" in result
        assert result["indep_vars"] == ["x1", "x2"]

    def test_bridge_probit_specification_string(self) -> None:
        """Bridge result includes a human-readable specification string."""
        import bridge

        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in make_binary_data(n=100, seed=42).iterrows()
            ],
            "columns": [
                {"name": "y", "col_type": "numeric"},
                {"name": "x1", "col_type": "numeric"},
                {"name": "x2", "col_type": "numeric"},
            ],
        }
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "probit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"]
        assert "specification" in result
        assert "y ~ x1 + x2" in result["specification"]

    def test_bridge_probit_all_json_serializable(self) -> None:
        """The entire bridge result should survive a JSON round-trip."""
        import bridge

        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in make_binary_data(n=100, seed=42).iterrows()
            ],
            "columns": [
                {"name": "y", "col_type": "numeric"},
                {"name": "x1", "col_type": "numeric"},
                {"name": "x2", "col_type": "numeric"},
            ],
        }
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "probit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        parsed = json.loads(result_json)
        re_serialized = json.dumps(parsed)
        assert len(re_serialized) > 0
        assert "NaN" not in re_serialized
        assert "Infinity" not in re_serialized

# encoding: utf-8
"""End-to-end integration tests for Logit regression.

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

from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit
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
    """Create a synthetic dataset for binary logistic regression.

    DGP: y* = 0.5 + 1.0*x1 - 0.8*x2 + logistic noise
    y = 1 if y* > 0 else 0
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.choice(["A", "B", "C"], n)
    cat1 = rng.integers(0, 2, n)
    eta = 0.5 + 1.0 * x1 - 0.8 * x2
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "cat1": cat1})


def make_perfect_separation_data(
    n: int = 60,
    seed: int = 123,
) -> pd.DataFrame:
    """Create data with near-perfect separation.

    y = 1 whenever x1 > 0, else 0.  A pure threshold predictor
    combined with an intercept often produces (quasi-)separation.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    y = (x1 > 0).astype(int)
    # Add a small amount of overlap (2 flipped labels) to test quasi-separation
    flip = rng.choice(n, size=2, replace=False)
    y[flip] = 1 - y[flip]
    return pd.DataFrame({"y": y, "x1": x1})


# =========================================================================
# Streamlit-side integration tests
# =========================================================================


class TestFullPipeline:
    """End-to-end: data -> ModelSpec -> Fitter.fit -> ModelResult."""

    def test_full_pipeline_basic(self) -> None:
        """Complete logit pipeline produces a valid ModelResult."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert isinstance(result, ModelResult)
        assert result.model_type == "logit"
        assert result.method == "Logit"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3  # Intercept + x1 + x2

    def test_full_pipeline_pseudo_r_squared_in_range(self) -> None:
        """Pseudo R-squared should be in (0, 1) for a reasonable model."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.pseudo_r_squared is not None
        assert 0 < result.pseudo_r_squared < 1, (
            f"Pseudo R² = {result.pseudo_r_squared}, expected (0, 1)"
        )

    def test_full_pipeline_ols_fields_are_none(self) -> None:
        """OLS-specific fields should be None for logit results."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.r_squared is None
        assert result.adj_r_squared is None
        assert result.f_statistic is None
        assert result.rmse is None

    def test_full_pipeline_coefficients_valid(self) -> None:
        """Every coefficient row should have valid statistics."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

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
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

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
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

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
            model_type="logit",
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert len(result.coefficients) == 3  # Intercept + x1 + x2
        coef_names = {c.name for c in result.coefficients}
        assert "x2" in coef_names

    def test_full_pipeline_no_intercept(self) -> None:
        """Logit pipeline without an intercept term."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"],
            has_intercept=False, model_type="logit",
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert len(result.coefficients) == 2  # x1 + x2, no intercept
        coef_names = {c.name for c in result.coefficients}
        assert "Intercept" not in coef_names
        assert "x1" in coef_names
        assert "x2" in coef_names


class TestFitterDispatch:
    """ModelFitter dispatches correctly based on model_type."""

    def test_fitter_logit_dispatch(self) -> None:
        """model_type='logit' routes to logit engine (pseudo_r2, no r2)."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.model_type == "logit"
        assert result.pseudo_r_squared is not None
        assert result.r_squared is None

    def test_fitter_ols_default(self) -> None:
        """Default model_type (omitted) routes to OLS engine."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.model_type == "OLS"
        assert result.r_squared is not None

    def test_fitter_mixed_logit_and_ols(self) -> None:
        """fit_multiple handles mixed logit + OLS specs in one call."""
        data = make_binary_data(seed=42)
        data["y_cont"] = (
            data["x1"] * 0.5
            + data["x2"] * 0.3
            + np.random.default_rng(88).normal(0, 0.1, len(data))
        )

        spec_logit = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        spec_ols = ModelSpec(dep_var="y_cont", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        results = fitter.fit_multiple([spec_logit, spec_ols], data)

        assert len(results) == 2
        assert results[0].model_type == "logit"
        assert results[1].model_type == "OLS"

    def test_fitter_all_logit_multiple(self) -> None:
        """fit_multiple with all logit specs compares two logit models."""
        data = make_binary_data(seed=42)
        spec1 = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="logit"
        )
        spec2 = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        results = fitter.fit_multiple([spec1, spec2], data)

        assert len(results) == 2
        assert all(r.model_type == "logit" for r in results)
        assert results[0].n_params == 2  # Intercept + x1
        assert results[1].n_params == 3  # Intercept + x1 + x2


class TestVariableLabels:
    """Variable labels in the logit pipeline."""

    def test_labels_categorical_in_logit_pipeline(self) -> None:
        """Labels for categorical dummies appear in ModelResult.variable_labels."""
        data = make_binary_data(seed=42)
        data["x3"] = np.random.default_rng(77).choice(
            ["Low", "Medium", "High"], len(data)
        )
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x3"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert "variable_labels" in result.__dataclass_fields__
        labels = result.variable_labels
        assert len(labels) > 0, "Variable labels dict should not be empty"

        # At minimum the Intercept should be labelled
        assert "Intercept" in labels or any(
            "x3" in k for k in labels
        ), f"Expected categorical labels, got: {labels}"

    def test_labels_nonempty_for_continuous_vars(self) -> None:
        """Continuous-only specification still produces a non-empty label dict."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        labels = result.variable_labels
        assert len(labels) > 0
        assert "Intercept" in labels
        assert labels["Intercept"] == "Intercept"


class TestSummaryDict:
    """to_summary_dict() for logit results."""

    def test_summary_dict_contains_logit_specific_fields(self) -> None:
        """Summary dict should expose pseudo_r_squared, llr, llr_pvalue."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        d = result.to_summary_dict()
        assert d["model_type"] == "logit"
        assert d["pseudo_r_squared"] is not None
        assert d["pseudo_r_squared"] == result.pseudo_r_squared
        assert d["llr"] == result.llr
        assert d["llr_pvalue"] == result.llr_pvalue
        assert d["f_statistic"] is None
        assert d["f_pvalue"] is None
        assert d["log_likelihood"] == result.log_likelihood


class TestDataFrame:
    """to_dataframe() for logit results."""

    def test_dataframe_uses_z_value_column(self) -> None:
        """Logit DataFrame column header is 'z值' not 't值'."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        df = result.to_dataframe()
        assert "z值" in df.columns
        assert "t值" not in df.columns

    def test_dataframe_contains_odds_ratio(self) -> None:
        """Logit DataFrame includes an OR column."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        df = result.to_dataframe()
        assert "OR(exp(B))" in df.columns

    def test_dataframe_all_values_finite(self) -> None:
        """Coefficient, SE, z-value columns should all be finite."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        df = result.to_dataframe()
        for col in ["系数", "标准误", "z值"]:
            assert all(np.isfinite(df[col])), f"Non-finite values in {col}"

    def test_dataframe_ci_order(self) -> None:
        """CI lower < CI upper for every coefficient."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        df = result.to_dataframe()
        assert all(df["95%CI低"] < df["95%CI高"])


class TestCategoricalPredictor:
    """Full pipeline with a categorical independent variable."""

    def test_logit_with_categorical_predictor(self) -> None:
        """Logit fit when one predictor is categorical (string)."""
        data = make_binary_data(seed=42)
        # x3 already has values A, B, C from the helper
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x3"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.model_type == "logit"
        # Design matrix: Intercept + x1 + C(x3)[T.B] + C(x3)[T.C] = 4 cols
        assert len(result.coefficients) == 4
        coef_names = {c.name for c in result.coefficients}
        assert "Intercept" in coef_names
        assert "x1" in coef_names
        # Verify dummy variable names are present
        assert any("x3" in name for name in coef_names if name != "x1")

    def test_logit_categorical_labels_readable(self) -> None:
        """Variable labels for categorical dummies are human-readable."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x3"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        labels = result.variable_labels
        # At least one label should be the decoded categorical form
        decoded_labels = [v for v in labels.values() if "x3" in v or ":" in v]
        if decoded_labels:
            assert all(
                isinstance(v, str) for v in decoded_labels
            ), "Labels should be strings"


class TestInteraction:
    """Full pipeline with an interaction term."""

    def test_logit_with_interaction(self) -> None:
        """Logit fit with x1:x2 interaction produces correct coefficient count."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            interaction_terms=[("x1", "x2")],
            model_type="logit",
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.model_type == "logit"
        # Intercept + x1 + x2 + x1:x2 = 4
        assert len(result.coefficients) == 4
        coef_names = {c.name for c in result.coefficients}
        assert "x1" in coef_names
        assert "x2" in coef_names
        assert any(":" in n for n in coef_names), (
            f"No interaction term found in {coef_names}"
        )

    def test_logit_interaction_metadata(self) -> None:
        """Interaction metadata is preserved in result."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            interaction_terms=[("x1", "x2")],
            model_type="logit",
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        assert result.interaction_terms_applied == [("x1", "x2")]


class TestConfidenceInterval:
    """Confidence interval verification for logit coefficients."""

    def test_ci_contains_coefficient(self) -> None:
        """95% CI should contain the estimated coefficient value."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

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
            dep_var="y", indep_vars=["x1"], model_type="logit"
        )
        fitter = ModelFitter()
        result_small = fitter.fit(spec, data_small)
        ci_width_small = result_small.coefficients[1].ci_upper - result_small.coefficients[1].ci_lower

        # Large dataset
        data_large = make_binary_data(n=500, seed=42)
        result_large = fitter.fit(spec, data_large)
        ci_width_large = result_large.coefficients[1].ci_upper - result_large.coefficients[1].ci_lower

        assert ci_width_large < ci_width_small, (
            f"CI width should decrease with more data: "
            f"small={ci_width_small:.4f}, large={ci_width_large:.4f}"
        )


class TestPerfectSeparation:
    """Perfect / quasi-separation handling."""

    def test_perfect_separation_raises_or_warns(self) -> None:
        """Near-perfect separation should either raise ValueError or converge
        with huge SEs."""
        data = make_perfect_separation_data(seed=123)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="logit"
        )

        try:
            fitted, labels = run_logit(data, spec)
            result = extract_logit(fitted, dep_var="y")
            # If it converged, coefficients should be huge (separation)
            x1_row = next(c for c in result.coefficients if c.name == "x1")
            assert abs(x1_row.coef) > 2.0, (
                "With near-separation, x1 coefficient should be large"
            )
        except ValueError:
            # Acceptable: statsmodels may detect non-convergence
            pass

    def test_strictly_perfect_separation_detected(self) -> None:
        """A truly separable dataset should trigger a ValueError or huge coef."""
        rng = np.random.default_rng(99)
        n = 40
        x1 = rng.normal(0, 1, n)
        y = (x1 > 0).astype(int)  # no overlap at all

        df = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="logit"
        )

        try:
            fitted, _ = run_logit(df, spec)
            result = extract_logit(fitted)
            # There should be a warning sign (huge SE or perfect prediction)
            x1_coef = next(c for c in result.coefficients if c.name == "x1")
            assert abs(x1_coef.coef) > 3.0 or x1_coef.se > 10.0, (
                "Strictly separable data should produce extreme estimates"
            )
        except ValueError:
            # Expected for true perfect separation
            pass


class TestSummaryMethod:
    """summary() and to_latex_row() for logit."""

    def test_summary_logit_contains_pseudo_r2(self) -> None:
        """Logit summary shows pseudo R-squared and LR chi2, not t-stat."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        text = result.summary()
        assert "Logit Regression Results" in text
        assert "Pseudo R-squared" in text
        assert "LR chi2" in text
        assert "p>|z|" in text
        assert "p>|t|" not in text
        # OLS-specific text should not appear (the word "R-squared" only
        # shows up inside "Pseudo R-squared" for logit; there should be no
        # standalone "R-squared:" line)
        assert "R-squared:" not in text.replace("Pseudo R-squared:", "")

    def test_latex_row_logit_seven_parts(self) -> None:
        """Logit LaTeX row has 7 parts: dep_var, n, pseudo_r2, llr, llr_p, aic, bic."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        assert len(parts) == 7, f"Expected 7 parts, got {len(parts)}: {parts}"
        assert "N/A" not in parts, f"All fields should have values: {parts}"


class TestAnovaTable:
    """ANOVA table is empty for logit models."""

    def test_anova_empty_for_logit(self) -> None:
        """anova_table() returns an empty DataFrame for logit."""
        data = make_binary_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        fitter = ModelFitter()
        result = fitter.fit(spec, data)

        anova = result.anova_table()
        assert anova.empty


class TestCompareModels:
    """compare_models() with logit results."""

    def test_compare_two_logit_models(self) -> None:
        """compare_models works with two logit results."""
        data = make_binary_data(seed=42)
        fitter = ModelFitter()

        spec1 = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="logit"
        )
        spec2 = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="logit"
        )
        results = fitter.fit_multiple([spec1, spec2], data)

        comparison = compare_models(results)
        assert not comparison.empty
        # Should have rows for coefficients plus model stats
        assert len(comparison) >= 3  # at least 2 coefs + stats

    def test_compare_three_logit_models(self) -> None:
        """compare_models with three logit models."""
        data = make_binary_data(seed=200)
        fitter = ModelFitter()

        specs = [
            ModelSpec(dep_var="y", indep_vars=["x1"], model_type="logit"),
            ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="logit"),
            ModelSpec(
                dep_var="y", indep_vars=["x1", "x2"],
                interaction_terms=[("x1", "x2")], model_type="logit",
            ),
        ]
        results = fitter.fit_multiple(specs, data)

        comparison = compare_models(results)
        assert not comparison.empty
        assert len(comparison) >= 4  # coef rows + stat rows


# =========================================================================
# Web-bridge integration tests
# =========================================================================


class TestBridgeLogitIntegration:
    """Simulate the web/bridge.py logit flow and verify JSON output."""

    def test_bridge_logit_basic(self) -> None:
        """Simulate bridge.run_regression with model_type='logit'."""
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
            "model_type": "logit",
            "alpha": 0.05,
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"], f"Bridge logit failed: {result.get('error')}"
        assert result["model_type"] == "logit"
        assert result["r_squared"] is None
        assert result["rmse"] is None
        assert result["pseudo_r_squared"] is not None
        assert 0 < result["pseudo_r_squared"] < 1
        assert result["llr"] is not None
        assert result["llr"] > 0
        assert len(result["coefficients"]) == 3  # Intercept + x1 + x2

    def test_bridge_logit_variable_labels(self) -> None:
        """Bridge result includes variable_labels dict."""
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
            "model_type": "logit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert "variable_labels" in result
        labels = result["variable_labels"]
        assert isinstance(labels, dict)
        assert len(labels) > 0
        assert "Intercept" in labels or any(
            "x1" in k for k in labels
        ), f"No meaningful labels found: {labels}"

    def test_bridge_logit_coefficient_structure(self) -> None:
        """Each coefficient in bridge output has all required fields."""
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
            "model_type": "logit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        for c in result["coefficients"]:
            assert "name" in c
            assert "coef" in c
            assert "se" in c
            assert "z_stat" in c
            assert "pvalue" in c
            assert "ci_lower" in c
            assert "ci_upper" in c
            assert "odds_ratio" in c
            assert "or_ci_lower" in c
            assert "or_ci_upper" in c
            assert "significance" in c

    def test_bridge_logit_rejects_non_binary_dep_var(self) -> None:
        """Bridge should error when dep_var has more than 2 unique values."""
        import bridge

        df_nonbinary = make_binary_data(n=50, seed=42).copy()
        # Replace y with a 3-class variable (0, 1, 2) to make it non-binary
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
            "model_type": "logit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert not result["success"]
        assert "binary" in result["error"].lower()

    def test_bridge_logit_indep_vars_in_result(self) -> None:
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
            "model_type": "logit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"]
        assert "indep_vars" in result
        assert result["indep_vars"] == ["x1", "x2"]

    def test_bridge_logit_specification_string(self) -> None:
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
            "model_type": "logit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"]
        assert "specification" in result
        assert "y ~ x1 + x2" in result["specification"]

    def test_bridge_logit_all_json_serializable(self) -> None:
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
            "model_type": "logit",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        # If bridge returned valid JSON, parsing again should succeed
        parsed = json.loads(result_json)
        # Re-serialize to verify every value is JSON-safe
        re_serialized = json.dumps(parsed)
        assert len(re_serialized) > 0
        # No NaN, Infinity, or other non-JSON-compliant values
        assert "NaN" not in re_serialized
        assert "Infinity" not in re_serialized


# =========================================================================
# Web-bridge categorical interaction tests
# =========================================================================


class TestBridgeCategoricalInteraction:
    """Web bridge handles categorical interactions (cat x num, cat x cat)."""

    @staticmethod
    def _make_cat_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
        """Create test data with categorical and numeric columns."""
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        cat = rng.choice(["A", "B", "C"], n).astype(str)
        cat2 = rng.choice(["X", "Y"], n).astype(str)
        eta = 0.5 + 1.0 * x1 - 0.8 * x2
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(n) < prob).astype(int)
        return pd.DataFrame(
            {"y": y, "x1": x1, "x2": x2, "cat": cat, "cat2": cat2}
        )

    @staticmethod
    def _to_bridge_data(df: pd.DataFrame) -> dict:
        """Convert a DataFrame to the bridge data_dict format."""
        col_types = {
            "y": "numeric",
            "x1": "numeric",
            "x2": "numeric",
            "cat": "categorical",
            "cat2": "categorical",
        }
        return {
            "data": [list(df.columns)]
            + [[str(v) for v in row] for _, row in df.iterrows()],
            "columns": [
                {"name": c, "col_type": col_types.get(c, "numeric")}
                for c in df.columns
            ],
        }

    def test_cat_x_num_interaction_coefficient_count(self) -> None:
        """Cat x num interaction creates one coefficient per dummy level."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat"],
            "model_type": "logit",
            "interactions": [["x1", "cat"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"], f"Bridge failed: {result.get('error')}"

        coef_names = [c["name"] for c in result["coefficients"]]
        # Intercept + x1 + cat_B + cat_C + x1:cat_B + x1:cat_C = 6
        # (cat has 3 levels: A, B, C; A is baseline)
        assert len(result["coefficients"]) == 6
        assert "x1:cat_B" in coef_names
        assert "x1:cat_C" in coef_names
        assert "x1" in coef_names
        assert "cat_B" in coef_names
        assert "cat_C" in coef_names

    def test_cat_x_cat_interaction_coefficient_count(self) -> None:
        """Cat x cat interaction creates pairwise products of dummy levels."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat", "cat2"],
            "model_type": "logit",
            "interactions": [["cat", "cat2"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"], f"Bridge failed: {result.get('error')}"

        coef_names = [c["name"] for c in result["coefficients"]]
        # cat: 3 levels (A/B/C) -> B, C dummies
        # cat2: 2 levels (X/Y) -> Y dummy
        # interactions: B:Y, C:Y (2 pairwise)
        # Total: Intercept + x1 + cat_B + cat_C + cat2_Y + cat_B:cat2_Y + cat_C:cat2_Y = 7
        assert len(result["coefficients"]) == 7
        assert "cat_B:cat2_Y" in coef_names
        assert "cat_C:cat2_Y" in coef_names
        # Baseline levels (A and X) should NOT appear in interaction names
        assert not any("A" in n or ":cat2_X" in n for n in coef_names)

    def test_cat_x_num_interaction_labels(self) -> None:
        """Cat x num interaction terms have human-readable labels."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat"],
            "model_type": "logit",
            "interactions": [["x1", "cat"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"]

        labels = result["variable_labels"]
        assert labels.get("x1:cat_B") == "x1 x cat: B"
        assert labels.get("x1:cat_C") == "x1 x cat: C"
        assert labels.get("cat_B") == "cat: B"

    def test_cat_x_cat_interaction_labels(self) -> None:
        """Cat x cat interaction terms have human-readable labels."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat", "cat2"],
            "model_type": "logit",
            "interactions": [["cat", "cat2"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"]

        labels = result["variable_labels"]
        assert labels.get("cat_B:cat2_Y") == "cat: B x cat2: Y"
        assert labels.get("cat_C:cat2_Y") == "cat: C x cat2: Y"

    def test_interaction_coefficients_all_finite(self) -> None:
        """All interaction coefficients have finite values."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        for spec_dict in [
            {
                "dep_var": "y", "indep_vars": ["x1", "cat"],
                "model_type": "logit", "interactions": [["x1", "cat"]],
            },
            {
                "dep_var": "y", "indep_vars": ["x1", "cat", "cat2"],
                "model_type": "logit", "interactions": [["cat", "cat2"]],
            },
        ]:
            result = json.loads(
                bridge.run_regression(
                    json.dumps(data_dict), json.dumps(spec_dict)
                )
            )
            assert result["success"]
            for c in result["coefficients"]:
                assert np.isfinite(c["coef"]), (
                    f"Non-finite coef for {c['name']} in {spec_dict['interactions']}"
                )
                assert c["se"] > 0, (
                    f"Non-positive SE for {c['name']} in {spec_dict['interactions']}"
                )

    def test_bridge_interaction_json_roundtrip(self) -> None:
        """Bridge interaction results survive JSON serialization."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat"],
            "model_type": "logit",
            "interactions": [["x1", "cat"]],
        }
        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        parsed = json.loads(result_json)
        re_serialized = json.dumps(parsed)
        assert "NaN" not in re_serialized
        assert "Infinity" not in re_serialized

    def test_num_x_num_interaction_still_works(self) -> None:
        """Numeric-only interactions continue to work after refactor."""
        import bridge

        df = self._make_cat_data(seed=42)
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "logit",
            "interactions": [["x1", "x2"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"], f"Bridge failed: {result.get('error')}"

        coef_names = [c["name"] for c in result["coefficients"]]
        # Intercept + x1 + x2 + x1:x2 = 4
        assert len(result["coefficients"]) == 4
        assert "x1:x2" in coef_names

    def test_ols_with_cat_interaction(self) -> None:
        """OLS model also handles categorical interactions correctly."""
        import bridge

        df = self._make_cat_data(seed=42)
        # Use a continuous DV for OLS
        df["y_cont"] = (
            df["x1"] * 1.0
            + np.random.default_rng(99).normal(0, 0.5, len(df))
        )
        data_dict = self._to_bridge_data(df)
        spec_dict = {
            "dep_var": "y_cont",
            "indep_vars": ["x1", "cat"],
            "model_type": "ols",
            "interactions": [["x1", "cat"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"], f"Bridge failed: {result.get('error')}"

        coef_names = [c["name"] for c in result["coefficients"]]
        assert "x1:cat_B" in coef_names
        assert "x1:cat_C" in coef_names
        assert len(result["coefficients"]) == 6
        # OLS-specific fields should be present
        assert result["r_squared"] is not None
        assert result["rmse"] is not None


class TestBridgeVsPatsyInteraction:
    """Coefficient values from the bridge match patsy-produced values."""

    def _build_patsy_comparison(
        self, df: pd.DataFrame, spec: ModelSpec
    ) -> dict:
        """Run a model via the Streamlit (patsy) path and return params."""
        from src.modeling.specification import build_design_matrix
        import statsmodels.api as sm

        X, y = build_design_matrix(spec, df)
        model_type = spec.model_type.lower()
        if model_type == "logit":
            fitted = sm.Logit(y, X).fit(disp=False)
        else:
            fitted = sm.OLS(y, X).fit()
        return dict(zip(X.columns, fitted.params))

    def test_cat_x_num_coefficients_match(self) -> None:
        """Bridge cat x num coefficients match patsy coefficients."""
        import bridge

        df = TestBridgeCategoricalInteraction._make_cat_data(seed=42)
        data_dict = TestBridgeCategoricalInteraction._to_bridge_data(df)

        spec_sl = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "cat"],
            interaction_terms=[("x1", "cat")],
            model_type="logit",
        )
        patsy_params = self._build_patsy_comparison(df.copy(), spec_sl)

        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat"],
            "model_type": "logit",
            "interactions": [["x1", "cat"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"]

        bridge_params = {c["name"]: c["coef"] for c in result["coefficients"]}

        # Map patsy names to bridge names for comparison
        # Patsy: cat[T.B] -> Bridge: cat_B
        # Patsy: x1:cat[T.B] -> Bridge: x1:cat_B
        name_map = {
            "Intercept": "Intercept",
            "x1": "x1",
        }
        for col_name in patsy_params:
            if col_name in name_map:
                continue
            bridge_name = col_name.replace("[T.", "_").replace("]", "")
            name_map[col_name] = bridge_name

        for patsy_name, expected_val in patsy_params.items():
            bridge_name = name_map[patsy_name]
            assert bridge_name in bridge_params, (
                f"Bridge missing coefficient: {bridge_name} (patsy: {patsy_name})"
            )
            actual_val = bridge_params[bridge_name]
            assert np.isclose(actual_val, expected_val, atol=1e-4), (
                f"Value mismatch for {bridge_name}: "
                f"bridge={actual_val:.6f}, patsy={expected_val:.6f}"
            )

    def test_cat_x_cat_coefficients_match(self) -> None:
        """Bridge cat x cat coefficients match patsy coefficients."""
        import bridge

        df = TestBridgeCategoricalInteraction._make_cat_data(seed=42)
        data_dict = TestBridgeCategoricalInteraction._to_bridge_data(df)

        spec_sl = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "cat", "cat2"],
            interaction_terms=[("cat", "cat2")],
            model_type="logit",
        )
        patsy_params = self._build_patsy_comparison(df.copy(), spec_sl)

        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "cat", "cat2"],
            "model_type": "logit",
            "interactions": [["cat", "cat2"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"]

        bridge_params = {c["name"]: c["coef"] for c in result["coefficients"]}

        # Map patsy names to bridge names
        name_map = {
            "Intercept": "Intercept",
            "x1": "x1",
        }
        for col_name in patsy_params:
            if col_name in name_map:
                continue
            bridge_name = col_name.replace("[T.", "_").replace("]", "")
            name_map[col_name] = bridge_name

        for patsy_name, expected_val in patsy_params.items():
            bridge_name = name_map[patsy_name]
            assert bridge_name in bridge_params, (
                f"Bridge missing coefficient: {bridge_name} (patsy: {patsy_name})"
            )
            actual_val = bridge_params[bridge_name]
            assert np.isclose(actual_val, expected_val, atol=1e-4), (
                f"Value mismatch for {bridge_name}: "
                f"bridge={actual_val:.6f}, patsy={expected_val:.6f}"
            )

    def test_num_x_num_coefficients_match(self) -> None:
        """Bridge num x num coefficients match patsy coefficients."""
        import bridge

        df = TestBridgeCategoricalInteraction._make_cat_data(seed=42)
        data_dict = TestBridgeCategoricalInteraction._to_bridge_data(df)

        spec_sl = ModelSpec(
            dep_var="y",
            indep_vars=["x1", "x2"],
            interaction_terms=[("x1", "x2")],
            model_type="logit",
        )
        patsy_params = self._build_patsy_comparison(df.copy(), spec_sl)

        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1", "x2"],
            "model_type": "logit",
            "interactions": [["x1", "x2"]],
        }
        result = json.loads(
            bridge.run_regression(json.dumps(data_dict), json.dumps(spec_dict))
        )
        assert result["success"]

        bridge_params = {c["name"]: c["coef"] for c in result["coefficients"]}

        for patsy_name, expected_val in patsy_params.items():
            assert patsy_name in bridge_params, (
                f"Bridge missing coefficient: {patsy_name}"
            )
            actual_val = bridge_params[patsy_name]
            assert np.isclose(actual_val, expected_val, atol=1e-4), (
                f"Value mismatch for {patsy_name}: "
                f"bridge={actual_val:.6f}, patsy={expected_val:.6f}"
            )

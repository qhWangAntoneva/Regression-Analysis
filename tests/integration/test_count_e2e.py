"""End-to-end integration tests for count regression (Poisson and NegBin).

Covers the complete pipeline from data preparation through model fitting
to result extraction, formatting, and export.

Tests are organised into two groups:
    - Direct engine: run_count_model -> extract_count_model -> formatting
    - Web-bridge: bridge.py data flow -> JSON-serializable output

NOTE: ModelFitter dispatch is NOT yet wired for count models (documented
in count_shared_changes.txt).  These tests use direct engine calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_count_engine import (
    _validate_count_dv,
    extract_count_model,
    run_count_model,
)
from src.modeling.specification import ModelSpec
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

def make_poisson_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create count data from a Poisson DGP."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.choice(["A", "B", "C"], n)
    lam = np.exp(1.0 + 0.3 * x1 - 0.15 * x2)
    y = rng.poisson(lam)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def make_overdispersed_data(n: int = 300, seed: int = 77) -> pd.DataFrame:
    """Create overdispersed count data (Var > Mean)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 1.0 + 0.5 * x1 - 0.3 * x2
    mu = np.exp(eta)
    size = 1.0
    prob = size / (size + mu)
    y = rng.negative_binomial(size, prob)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# =========================================================================
# Direct engine end-to-end tests
# =========================================================================


class TestPoissonE2E:
    """End-to-end: data -> ModelSpec -> run_count_model -> extract_count_model."""

    def test_full_pipeline_poisson(self) -> None:
        """Complete Poisson pipeline produces a valid ModelResult."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, labels = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        assert isinstance(result, ModelResult)
        assert result.model_type == "poisson"
        assert result.method == "Poisson"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3  # Intercept + x1 + x2

    def test_poisson_pseudo_r_squared(self) -> None:
        """Pseudo R-squared should be in (0, 1) for a reasonable model."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.pseudo_r_squared is not None
        assert 0 < result.pseudo_r_squared < 1, (
            f"Pseudo R² = {result.pseudo_r_squared}, expected (0, 1)"
        )

    def test_poisson_ols_fields_are_none(self) -> None:
        """OLS-specific fields should be None for Poisson results."""
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

    def test_poisson_coefficients_valid(self) -> None:
        """Every coefficient row should have valid statistics."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

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

    def test_poisson_log_likelihood_and_llr(self) -> None:
        """Log-likelihood and LLR should be finite."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        assert result.log_likelihood is not None
        assert np.isfinite(result.log_likelihood)
        assert result.llr is not None
        assert result.llr > 0

    def test_poisson_dep_var_and_specification(self) -> None:
        """The ModelResult should preserve dep_var and specification."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        spec_str = "y ~ x1 + x2"
        result = extract_count_model(
            fitted, dep_var="y", specification=spec_str
        )

        assert result.dep_var == "y"
        assert "x1" in result.specification
        assert "x2" in result.specification


class TestNegBinE2E:
    """End-to-end: data -> ModelSpec -> run_count_model -> extract_count_model."""

    def test_full_pipeline_negbin(self) -> None:
        """Complete NegBin pipeline produces a valid ModelResult."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, labels = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        assert isinstance(result, ModelResult)
        assert result.model_type == "negbin"
        assert result.method == "NegativeBinomial"
        assert result.n_obs > 0
        assert len(result.coefficients) == 3

    def test_negbin_dispersion_preserved(self) -> None:
        """Dispersion parameter should be accessible from the fitted model."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)

        # Verify dispersion (scale) is available and positive
        scale = getattr(fitted, "scale", None)
        assert scale is not None, "NB fitted model should have scale attribute"
        assert scale > 0, f"Expected positive dispersion, got {scale}"

        # Also verify it is finite
        assert np.isfinite(scale)

    def test_negbin_coefficients_valid(self) -> None:
        """Every coefficient row should have valid statistics for NegBin."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        for c in result.coefficients:
            assert isinstance(c, CoefficientRow)
            assert isinstance(c.name, str) and len(c.name) > 0
            assert np.isfinite(c.coef)
            assert c.se > 0
            assert np.isfinite(c.t_stat)
            assert 0 <= c.pvalue <= 1
            assert c.ci_lower < c.ci_upper


# =========================================================================
# Count data validation
# =========================================================================

class TestCountDataValidation:
    """Validation of count-data requirements."""

    def test_non_integer_dv_error_message(self) -> None:
        """Non-integer DV should produce a clear error message."""
        n = 50
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 1, n)
        y = rng.uniform(0, 10, n)
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="poisson"
        )

        with pytest.raises(ValueError, match="integer"):
            run_count_model(data, spec)

    def test_negative_dv_error_message(self) -> None:
        """Negative DV should produce a clear error message."""
        n = 50
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 1, n)
        y = rng.integers(-3, 5, n)
        data = pd.DataFrame({"y": y, "x1": x1})
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="negbin"
        )

        with pytest.raises(ValueError, match="negative"):
            run_count_model(data, spec)

    def test_validate_count_dv_function(self) -> None:
        """Direct test of _validate_count_dv helper."""
        # Valid: non-negative integers
        y = pd.Series([0, 1, 2, 3, 5, 10], name="y")
        _validate_count_dv(y, "Poisson")  # Should not raise

        # Invalid: negative
        y_neg = pd.Series([-1, 0, 1, 2], name="y")
        with pytest.raises(ValueError, match="negative"):
            _validate_count_dv(y_neg, "Poisson")

        # Invalid: non-integer
        y_float = pd.Series([0.5, 1.0, 2.3], name="y")
        with pytest.raises(ValueError, match="integer"):
            _validate_count_dv(y_float, "NegativeBinomial")


# =========================================================================
# Export / formatting pipeline
# =========================================================================

class TestExportPipeline:
    """Export and formatting for count models."""

    def test_poisson_to_dataframe_export(self) -> None:
        """Poisson results can be exported to DataFrame."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        df = result.to_dataframe()
        assert not df.empty
        assert len(df) == len(result.coefficients)
        assert "z值" in df.columns

    def test_negbin_to_dataframe_export(self) -> None:
        """NegBin results can be exported to DataFrame."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        df = result.to_dataframe()
        assert not df.empty
        assert len(df) == len(result.coefficients)
        assert "z值" in df.columns

    def test_poisson_summary_dict_export(self) -> None:
        """Poisson summary dict is JSON-serializable."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        d = result.to_summary_dict()
        # Should survive JSON round-trip
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "poisson"
        assert parsed["pseudo_r_squared"] is not None

    def test_negbin_summary_dict_export(self) -> None:
        """NegBin summary dict is JSON-serializable."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        d = result.to_summary_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["model_type"] == "negbin"
        assert parsed["pseudo_r_squared"] is not None

    def test_poisson_to_latex_row(self) -> None:
        """LaTeX row for Poisson has 7 parts (MLE format)."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        assert len(parts) == 7, f"Expected 7 parts, got {len(parts)}: {parts}"
        assert "N/A" not in parts, f"All fields should have values: {parts}"

    def test_negbin_to_latex_row(self) -> None:
        """LaTeX row for NegBin has 7 parts (MLE format)."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        assert len(parts) == 7, f"Expected 7 parts, got {len(parts)}: {parts}"


# =========================================================================
# compare_models with count models
# =========================================================================

class TestCompareModels:
    """compare_models() with count model results."""

    def test_compare_two_poisson_models(self) -> None:
        """compare_models works with two Poisson results."""
        data = make_poisson_data(seed=42)

        spec1 = ModelSpec(
            dep_var="y", indep_vars=["x1"], model_type="poisson"
        )
        spec2 = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )

        fitted1, _ = run_count_model(data, spec1)
        fitted2, _ = run_count_model(data, spec2)
        result1 = extract_count_model(fitted1, dep_var="y")
        result2 = extract_count_model(fitted2, dep_var="y")

        comparison = compare_models([result1, result2])
        assert not comparison.empty
        assert len(comparison) >= 3  # at least 2 coefs + stats

    def test_compare_poisson_vs_negbin(self) -> None:
        """compare_models works with Poisson vs NegBin results."""
        data = make_overdispersed_data(seed=77)

        poisson_spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        negbin_spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )

        p_fitted, _ = run_count_model(data, poisson_spec)
        nb_fitted, _ = run_count_model(data, negbin_spec)
        poisson_result = extract_count_model(p_fitted, dep_var="y")
        negbin_result = extract_count_model(nb_fitted, dep_var="y")

        comparison = compare_models([poisson_result, negbin_result])
        assert not comparison.empty
        assert len(comparison) >= 3

    def test_compare_three_count_models(self) -> None:
        """compare_models with three count models."""
        data = make_poisson_data(seed=200)

        specs = [
            ModelSpec(dep_var="y", indep_vars=["x1"], model_type="poisson"),
            ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"),
            ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"),
        ]

        results = []
        for spec in specs:
            fitted, _ = run_count_model(data, spec)
            result = extract_count_model(fitted, dep_var="y")
            results.append(result)

        comparison = compare_models(results)
        assert not comparison.empty
        assert len(comparison) >= 4  # coef rows + stat rows


# =========================================================================
# ANOVA table (should be empty for count models)
# =========================================================================

class TestAnovaTable:
    """ANOVA table is empty for count models."""

    def test_anova_empty_for_poisson(self) -> None:
        """anova_table() returns empty DataFrame for Poisson."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        anova = result.anova_table()
        assert anova.empty

    def test_anova_empty_for_negbin(self) -> None:
        """anova_table() returns empty DataFrame for NegBin."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        anova = result.anova_table()
        assert anova.empty


# =========================================================================
# Variable labels in the pipeline
# =========================================================================

class TestVariableLabels:
    """Variable labels in the count model pipeline."""

    def test_labels_include_intercept(self) -> None:
        """All count models should include an Intercept label."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, labels = run_count_model(data, spec)
        result = extract_count_model(fitted, variable_labels=labels)

        assert "Intercept" in result.variable_labels
        assert result.variable_labels["Intercept"] == "Intercept"

    def test_labels_nonempty_continuous_vars(self) -> None:
        """Continuous-only specification still produces non-empty labels."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, labels = run_count_model(data, spec)
        result = extract_count_model(fitted, variable_labels=labels)

        assert len(result.variable_labels) > 0
        assert "Intercept" in result.variable_labels


# =========================================================================
# Summary method for count models
# =========================================================================

class TestSummaryMethod:
    """summary() and to_latex_row() for count models."""

    def test_summary_poisson_contains_pseudo_r2(self) -> None:
        """Poisson summary shows pseudo R-squared and z-stats."""
        data = make_poisson_data(seed=42)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="poisson"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted, dep_var="y")

        text = result.summary()
        assert "Poisson Regression Results" in text
        assert "Pseudo R-squared" in text
        assert "p>|z|" in text
        assert "p>|t|" not in text

    def test_summary_negbin_contains_method(self) -> None:
        """NegBin summary shows the method name."""
        data = make_overdispersed_data(seed=77)
        spec = ModelSpec(
            dep_var="y", indep_vars=["x1", "x2"], model_type="negbin"
        )
        fitted, _ = run_count_model(data, spec)
        result = extract_count_model(fitted)

        text = result.summary()
        assert "NegativeBinomial Regression Results" in text


# =========================================================================
# Web-bridge integration tests (simulated)
# =========================================================================

class TestBridgeCountIntegration:
    """Simulate the web/bridge.py count-model flow and verify JSON output."""

    @pytest.mark.skip(reason="bridge.py not yet wired for count models — documented in count_shared_changes.txt")  # noqa: E501
    def test_bridge_poisson_basic(self) -> None:
        """Simulate bridge.run_regression with model_type='poisson'."""
        import bridge

        data = make_poisson_data(n=100, seed=42)
        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in data.iterrows()
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
            "model_type": "poisson",
            "alpha": 0.05,
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"], f"Bridge Poisson failed: {result.get('error')}"
        assert result["model_type"] == "poisson"
        assert result["r_squared"] is None
        assert result["rmse"] is None
        assert result["pseudo_r_squared"] is not None
        assert len(result["coefficients"]) == 3  # Intercept + x1 + x2

    @pytest.mark.skip(reason="bridge.py not yet wired for count models — documented in count_shared_changes.txt")  # noqa: E501
    def test_bridge_negbin_basic(self) -> None:
        """Simulate bridge.run_regression with model_type='negbin'."""
        import bridge

        data = make_overdispersed_data(n=150, seed=77)
        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in data.iterrows()
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
            "model_type": "negbin",
            "alpha": 0.05,
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert result["success"], f"Bridge NegBin failed: {result.get('error')}"
        assert result["model_type"] == "negbin"
        assert result["r_squared"] is None
        assert result["pseudo_r_squared"] is not None

    def test_bridge_poisson_rejects_negative_dv(self) -> None:
        """Bridge should error when Poisson DV has negative values."""
        import bridge

        n = 50
        rng = np.random.default_rng(42)
        data_dict = {
            "data": [
                ["y", "x1"],
            ]
            + [
                [str(rng.integers(-1, 5)), str(rng.normal(0, 1))]
                for _ in range(n)
            ],
            "columns": [
                {"name": "y", "col_type": "numeric"},
                {"name": "x1", "col_type": "numeric"},
            ],
        }
        spec_dict = {
            "dep_var": "y",
            "indep_vars": ["x1"],
            "model_type": "poisson",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)

        assert not result["success"]

    @pytest.mark.skip(reason="bridge.py not yet wired for count models — documented in count_shared_changes.txt")  # noqa: E501
    def test_bridge_poisson_coefficient_structure(self) -> None:
        """Each coefficient in bridge output has all required fields."""
        import bridge

        data = make_poisson_data(n=100, seed=42)
        data_dict = {
            "data": [
                ["y", "x1", "x2"],
            ]
            + [
                [str(int(row.y)), str(row.x1), str(row.x2)]
                for _, row in data.iterrows()
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
            "model_type": "poisson",
        }

        result_json = bridge.run_regression(
            json.dumps(data_dict), json.dumps(spec_dict)
        )
        result = json.loads(result_json)
        assert result["success"]

        for c in result["coefficients"]:
            assert "name" in c
            assert "coef" in c
            assert "se" in c
            assert "z_stat" in c
            assert "pvalue" in c
            assert "ci_lower" in c
            assert "ci_upper" in c
            assert "significance" in c

"""End-to-end integration tests for MixedLM (multilevel) regression.

Covers the complete pipeline from data preparation through model fitting
to result extraction, formatting, and cross-model comparison.

Tests are organised into:
    - Engine pipeline: data -> MixedLM -> ModelResult -> formatting
    - Comparison: MixedLM vs OLS on clustered data
    - Export: to_dataframe(), to_summary_dict(), to_latex_row()
    - Comparison tables: compare_models() with MixedLM
    - Web-bridge: result serialisation compatibility
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_engine import run_ols
from src.modeling.engines.statsmodels_mixedlm_engine import (
    extract_mixedlm,
    run_and_extract_mixedlm,
    run_mixedlm,
)
from src.modeling.specification import ModelSpec
from src.results.table import ModelResult, compare_models

# -----------------------------------------------------------------------
# Make the web bridge importable for integration testing
# -----------------------------------------------------------------------
_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "py"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))


# =========================================================================
# Helpers
# =========================================================================


def make_grouped_data(
    n_groups: int = 20,
    n_per_group: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic grouped dataset with random intercepts.

    DGP: y = 2.0 + 1.5*x1 - 0.8*x2 + group_effect + noise
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


def make_clustered_data_unbalanced(
    seed: int = 123,
) -> pd.DataFrame:
    """Create unbalanced clustered data (different group sizes)."""
    rng = np.random.default_rng(seed)
    group_sizes = [5, 8, 12, 15, 10, 20, 7, 14, 9, 11]
    group_ids = []
    for gid, size in enumerate(group_sizes):
        group_ids.extend([gid] * size)
    group_ids = np.array(group_ids)
    total_n = len(group_ids)
    x1 = rng.normal(0, 1, total_n)
    x2 = rng.normal(0, 1, total_n)
    group_effects = rng.normal(0, 0.4, len(group_sizes))
    y = (
        1.0
        + 0.9 * x1
        - 0.5 * x2
        + group_effects[group_ids]
        + rng.normal(0, 0.3, total_n)
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "school": group_ids})


# =========================================================================
# Tests: Full engine pipeline
# =========================================================================


class TestEnginePipeline:
    """End-to-end: data -> MixedLM -> ModelResult."""

    def test_full_pipeline(self) -> None:
        """Complete pipeline produces correct ModelResult."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"

        fitted, labels = run_mixedlm(df, spec)
        result = extract_mixedlm(
            fitted,
            dep_var="y",
            specification="y ~ x1 + x2 + (1 | group)",
            variable_labels=labels,
        )

        assert isinstance(result, ModelResult)
        assert result.model_type == "mixedlm"
        assert result.method == "MixedLM (REML)"
        assert result.n_obs == 300
        assert len(result.coefficients) == 3

    def test_pipeline_with_control_variables(self) -> None:
        """Pipeline handles independent + control variables."""
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
        assert len(result.coefficients) == 3

    def test_unbalanced_groups(self) -> None:
        """Pipeline works with unbalanced group sizes."""
        df = make_clustered_data_unbalanced()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "school"

        result = run_and_extract_mixedlm(df, spec)

        assert result.mixedlm_converged
        assert result.group_count == 10
        assert result.r_squared is not None
        assert result.r_squared > 0

    def test_csv_with_group_column(self) -> None:
        """End-to-end with CSV input containing a group column."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "group"

        result = run_and_extract_mixedlm(df, spec)

        assert result.n_obs == len(df)
        assert result.group_count == 20
        assert len(result.coefficients) == 2  # Intercept + x1


# =========================================================================
# Tests: Comparison MixedLM vs OLS
# =========================================================================


class TestMixedLMvsOLS:
    """Compare MixedLM and OLS on clustered data."""

    def test_se_differ_from_ols(self) -> None:
        """MixedLM SE should differ from OLS SE on clustered data.

        OLS treats all observations as independent, so its SEs are
        typically too small when observations are clustered.  MixedLM
        correctly accounts for clustering, yielding larger SEs for
        variables that vary within groups.
        """
        df = make_grouped_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"

        # Fit MixedLM
        mixed_result = run_and_extract_mixedlm(df, spec)

        # Fit OLS on same data (ignoring clustering)
        ols_spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        ols_result = run_ols(df, ols_spec)

        # Compare SEs for x1 (within-group predictor)
        mixed_coefs = {c.name: c for c in mixed_result.coefficients}
        ols_coefs = {c.name: c for c in ols_result.coefficients}

        # SEs should differ at least slightly
        assert mixed_coefs["x1"].se != pytest.approx(ols_coefs["x1"].se, rel=0.001), (
            f"MixedLM SE ({mixed_coefs['x1'].se:.6f}) should differ from "
            f"OLS SE ({ols_coefs['x1'].se:.6f}) on clustered data"
        )

    def test_coefficients_in_same_direction(self) -> None:
        """MixedLM and OLS coefficients should have the same sign."""
        df = make_grouped_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"

        mixed_result = run_and_extract_mixedlm(df, spec)
        ols_spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        ols_result = run_ols(df, ols_spec)

        mixed_coefs = {c.name: c.coef for c in mixed_result.coefficients}
        ols_coefs = {c.name: c.coef for c in ols_result.coefficients}

        for name in ["x1", "x2", "Intercept"]:
            assert (mixed_coefs[name] > 0) == (ols_coefs[name] > 0), (
                f"{name}: MixedLM sign ({mixed_coefs[name]:.4f}) != "
                f"OLS sign ({ols_coefs[name]:.4f})"
            )

    def test_re_variance_from_clustered_data(self) -> None:
        """RE variance should capture group-level variation."""
        df = make_grouped_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"

        result = run_and_extract_mixedlm(df, spec)

        # Group Var should be roughly 0.25 (0.5^2 from DGP)
        re_var = result.re_var.get("Group Var", 0)
        assert re_var > 0.1, f"RE variance too small: {re_var}"
        assert re_var < 1.0, f"RE variance too large: {re_var}"


# =========================================================================
# Tests: Export pipeline
# =========================================================================


class TestExportPipeline:
    """Export pipeline: result -> DataFrame / dict / LaTeX."""

    def test_to_dataframe_all_finite(self) -> None:
        """All numeric columns in to_dataframe() should be finite."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        out = result.to_dataframe()
        for col in ["系数", "标准误", "t值"]:
            assert all(np.isfinite(out[col])), f"Non-finite values in {col}"

    def test_to_summary_dict_keys(self) -> None:
        """to_summary_dict() should have expected OLS-like keys."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        d = result.to_summary_dict()
        expected_keys = [
            "dep_var", "n_obs", "n_params", "df_resid",
            "r_squared", "adj_r_squared", "rmse",
            "log_likelihood", "aic", "bic",
            "method", "specification", "model_type",
            "pseudo_r_squared", "llr", "llr_pvalue",
            "f_statistic", "f_pvalue",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key '{key}' in to_summary_dict()"

    def test_to_latex_row_finite_values(self) -> None:
        """to_latex_row() for MixedLM: R² values present, f-stat N/A, AIC/BIC NaN.

        MixedLM has no F-test and REML AIC/BIC are NaN, so the LaTeX row
        shows N/A for f-stat and nan for AIC/BIC.  This is expected behaviour
        for REML-fitted mixed models.
        """
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        latex = result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        # dep_var & n & r2 & adj_r2 & f & fp & aic & bic
        assert len(parts) == 8
        assert result.r_squared is not None
        assert parts[2] != "N/A", f"R² should be present, got: {parts[2]}"
        # f-stat and f-pval are N/A (MixedLM has no F-test)
        assert parts[4] == "N/A"
        assert parts[5] == "N/A"


# =========================================================================
# Tests: Model comparison table
# =========================================================================


class TestModelComparison:
    """compare_models() with MixedLM."""

    def test_compare_mixedlm_with_ols(self) -> None:
        """compare_models() handles MixedLM + OLS side by side."""
        df = make_grouped_data()
        spec1 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        spec2 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec2.group_var = "group"

        ols_result = run_ols(df, spec1)
        mixed_result = run_and_extract_mixedlm(df, spec2)

        comp = compare_models([ols_result, mixed_result])
        assert not comp.empty
        assert len(comp) >= 3  # at least Intercept + x1 + x2 + stats rows
        assert "变量" in comp.columns

    def test_compare_multiple_mixedlm(self) -> None:
        """compare_models() handles multiple MixedLM specs."""
        df = make_grouped_data()
        spec1 = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec1.group_var = "group"
        spec2 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec2.group_var = "group"

        r1 = run_and_extract_mixedlm(df, spec1)
        r2 = run_and_extract_mixedlm(df, spec2)

        comp = compare_models([r1, r2])
        assert not comp.empty
        # x2 appears only in model 2 -- should have empty cell in model 1
        x2_row = comp[comp["变量"] == "x2"]
        assert not x2_row.empty

    def test_compare_three_models(self) -> None:
        """compare_models() with three MixedLM specs."""
        df = make_grouped_data()
        specs = [
            ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm"),
            ModelSpec(dep_var="y", indep_vars=["x2"], model_type="mixedlm"),
            ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm"),
        ]
        for s in specs:
            s.group_var = "group"

        results = [run_and_extract_mixedlm(df, s) for s in specs]
        comp = compare_models(results)

        assert not comp.empty
        assert len(comp.columns) >= 4  # 变量 + 3 models


# =========================================================================
# Tests: Error cases (integration)
# =========================================================================


class TestErrorCases:
    """Integration-level error handling for MixedLM."""

    def test_nonexistent_group_column(self) -> None:
        """Should raise when group column does not exist in data."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = "not_a_column"

        with pytest.raises(ValueError, match="not_a_column"):
            run_and_extract_mixedlm(df, spec)

    def test_group_var_none(self) -> None:
        """Should raise when group_var is None."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1"], model_type="mixedlm")
        spec.group_var = None

        with pytest.raises(ValueError, match="group_var"):
            run_and_extract_mixedlm(df, spec)


# =========================================================================
# Tests: Web bridge compatibility
# =========================================================================


class TestWebBridge:
    """MixedLM result serialisation for web bridge."""

    def test_to_summary_dict_json_serializable(self) -> None:
        """to_summary_dict() output should be JSON-serializable."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        d = result.to_summary_dict()
        # Should not raise
        json_str = json.dumps(d, allow_nan=True)
        assert len(json_str) > 0

    def test_coefficient_list_serializable(self) -> None:
        """Coefficient list should be convertible to list of dicts."""
        df = make_grouped_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], model_type="mixedlm")
        spec.group_var = "group"
        result = run_and_extract_mixedlm(df, spec)

        coef_list = []
        for c in result.coefficients:
            coef_list.append({
                "name": c.name,
                "coef": c.coef,
                "se": c.se,
                "t_stat": c.t_stat,
                "pvalue": c.pvalue,
                "ci_lower": c.ci_lower,
                "ci_upper": c.ci_upper,
                "significance": c.significance,
            })

        json_str = json.dumps(coef_list)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert len(parsed) == len(result.coefficients)

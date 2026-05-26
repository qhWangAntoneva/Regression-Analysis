# encoding: utf-8
"""End-to-end integration tests for Panel data regression.

Covers the complete pipeline from data preparation through panel model
fitting to result extraction, comparison, and output formatting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_panel_engine import extract_panel, run_panel
from src.modeling.specification import ModelSpec, build_variable_labels
from src.results.table import CoefficientRow, ModelResult, compare_models


# =========================================================================
# Helpers
# =========================================================================


def make_panel_data(
    n_entities: int = 30,
    n_periods: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic balanced panel data.

    DGP: y = 1.0 + 2.0*x1 - 1.5*x2 + entity_effect_i + noise
    """
    rng = np.random.default_rng(seed)
    entities = np.repeat(np.arange(n_entities), n_periods)
    times = np.tile(np.arange(n_periods), n_entities)
    entity_effects = rng.normal(0, 0.8, n_entities)
    x1 = rng.normal(0, 0.5, n_entities * n_periods)
    x2 = rng.normal(0, 0.3, n_entities * n_periods)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + entity_effects[entities] + rng.normal(0, 0.2, n_entities * n_periods)
    return pd.DataFrame({
        "entity": entities,
        "time": times,
        "y": y,
        "x1": x1,
        "x2": x2,
        "z": rng.normal(0, 0.4, n_entities * n_periods),
    })


def make_spec(entity_var: str, time_var: str, panel_model: str, **kwargs) -> ModelSpec:
    """Build a ModelSpec with panel attributes monkey-patched on."""
    spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True, **kwargs)
    spec.entity_var = entity_var       # type: ignore[attr-defined]
    spec.time_var = time_var          # type: ignore[attr-defined]
    spec.panel_model = panel_model    # type: ignore[attr-defined]
    return spec


# =========================================================================
# Tests: End-to-end Fixed Effects
# =========================================================================


class TestFE_EndToEnd:
    """End-to-end panel FE pipeline."""

    def test_fe_full_pipeline_basic(self) -> None:
        """Complete FE pipeline produces a valid ModelResult."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "fixed")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", specification="y ~ x1 + x2",
                               variable_labels=labels)

        assert isinstance(result, ModelResult)
        assert result.model_type == "panel"
        assert result.method == "Panel FE"
        assert result.n_obs == 240
        assert len(result.coefficients) >= 2
        assert result.within_r_squared is not None
        assert result.within_r_squared > 0

    def test_fe_result_to_dataframe(self) -> None:
        """FE ModelResult.to_dataframe() should produce a non-empty DataFrame."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "fixed")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", specification="y ~ x1 + x2")

        df = result.to_dataframe()
        assert not df.empty
        assert "系数" in df.columns
        assert "标准误" in df.columns
        assert "t值" in df.columns  # Panel is not MLE, uses t-statistic

    def test_fe_summary_dict(self) -> None:
        """FE ModelResult.to_summary_dict() should include model stats."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "fixed")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        d = result.to_summary_dict()
        assert d["model_type"] == "panel"
        assert d["method"] == "Panel FE"
        assert d["n_obs"] == 240
        assert d["r_squared"] is not None and d["r_squared"] > 0

    def test_fe_summary_output_nonempty(self) -> None:
        """FE summary() should contain key information."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "fixed")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", specification="y ~ x1 + x2")

        s = result.summary()
        assert "Panel FE" in s
        assert "y" in s.lower()
        assert "x1" in s
        assert "x2" in s


# =========================================================================
# Tests: End-to-end Random Effects
# =========================================================================


class TestRE_EndToEnd:
    """End-to-end panel RE pipeline."""

    def test_re_full_pipeline_basic(self) -> None:
        """Complete RE pipeline produces a valid ModelResult."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "random")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", specification="y ~ x1 + x2",
                               variable_labels=labels)

        assert isinstance(result, ModelResult)
        assert result.model_type == "panel"
        assert result.method == "Panel RE"
        assert result.n_obs == 240
        assert len(result.coefficients) >= 2
        assert result.overall_r_squared is not None

    def test_re_result_to_dataframe(self) -> None:
        """RE ModelResult.to_dataframe() should produce a non-empty DataFrame."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "random")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", specification="y ~ x1 + x2")

        df = result.to_dataframe()
        assert not df.empty
        assert "系数" in df.columns

    def test_re_summary_dict(self) -> None:
        """RE ModelResult.to_summary_dict() should include model stats."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "random")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        d = result.to_summary_dict()
        assert d["model_type"] == "panel"
        assert d["method"] == "Panel RE"

    def test_re_rmse_present(self) -> None:
        """RE result should have RMSE computed."""
        data = make_panel_data(seed=42)
        spec = make_spec("entity", "time", "random")

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.rmse is not None
        assert result.rmse > 0


# =========================================================================
# Tests: FE vs RE comparison
# =========================================================================


class TestComparison:
    """FE vs RE comparison on the same dataset."""

    def test_fe_vs_re_same_data(self) -> None:
        """FE and RE fitted on same data both produce valid results."""
        data = make_panel_data(seed=42)

        spec_fe = make_spec("entity", "time", "fixed")
        spec_re = make_spec("entity", "time", "random")

        fe_fitted, fe_labels = run_panel(data, spec_fe)
        re_fitted, re_labels = run_panel(data, spec_re)

        fe_result = extract_panel(fe_fitted, dep_var="y", specification="FE: y ~ x1 + x2")
        re_result = extract_panel(re_fitted, dep_var="y", specification="RE: y ~ x1 + x2")

        # Both should have valid results
        assert fe_result.n_obs == re_result.n_obs
        assert fe_result.entity_count == re_result.entity_count
        assert fe_result.time_count == re_result.time_count

        # Within R² for FE > RE within R² is common
        # (but not always guaranteed; just check they are computed)
        assert fe_result.within_r_squared is not None
        assert re_result.within_r_squared is not None

        # Coefficients should be similar but not identical
        fe_x1 = [c.coef for c in fe_result.coefficients if c.name == "x1"][0]
        re_x1 = [c.coef for c in re_result.coefficients if c.name == "x1"][0]
        assert abs(fe_x1 - re_x1) < 1.0, (
            f"FE and RE x1 coefficients differ too much: {fe_x1} vs {re_x1}"
        )

    def test_compare_models_function(self) -> None:
        """compare_models() should work with panel results."""
        data = make_panel_data(seed=42)

        spec_fe = make_spec("entity", "time", "fixed")
        spec_re = make_spec("entity", "time", "random")

        fe_fitted, _ = run_panel(data, spec_fe)
        re_fitted, _ = run_panel(data, spec_re)

        fe_result = extract_panel(fe_fitted, dep_var="y",
                                   specification="y ~ x1 + x2 [FE]")
        re_result = extract_panel(re_fitted, dep_var="y",
                                   specification="y ~ x1 + x2 [RE]")

        # compare_models should produce a non-empty DataFrame
        df = compare_models([fe_result, re_result])
        assert not df.empty
        assert "变量" in df.columns

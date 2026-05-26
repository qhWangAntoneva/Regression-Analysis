"""Unit tests for the Panel data regression engine (FE and RE).

Tests cover:
    - Fixed Effects: basic fitting, coefficient accuracy, within R-squared,
      standard errors, entity/time counts, t-statistics
    - Random Effects: basic fitting, between/overall R-squared
    - Error handling: missing entity_var, missing time_var, single entity
    - Metadata: model_type, panel_model, variable labels, CI bounds
    - F-pooled test for FE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.engines.statsmodels_panel_engine import extract_panel, run_panel
from src.modeling.specification import ModelSpec
from src.results.table import ModelResult

# =========================================================================
# Helper: generate balanced panel data with entity fixed effects
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
    return pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})


# =========================================================================
# Tests: Fixed Effects
# =========================================================================


class TestFixedEffects:
    """Fixed Effects panel model tests."""

    def test_fe_basic_fit(self) -> None:
        """FE should fit without error and return a valid result."""
        data = make_panel_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"      # type: ignore[attr-defined]
        spec.time_var = "time"          # type: ignore[attr-defined]
        spec.panel_model = "fixed"      # type: ignore[attr-defined]

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", variable_labels=labels)

        assert isinstance(result, ModelResult)
        assert result.model_type == "panel"
        assert len(result.coefficients) >= 2  # x1, x2 (intercept may be present)
        assert result.n_obs > 0

    def test_fe_coefficients_close_to_true(self) -> None:
        """FE coefficients should be close to DGP values (2.0 and -1.5)."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        coef_map = {c.name: c for c in result.coefficients}

        # x1 true value = 2.0
        assert "x1" in coef_map
        assert abs(coef_map["x1"].coef - 2.0) < 0.5, f"x1 coef={coef_map['x1'].coef}"

        # x2 true value = -1.5
        assert "x2" in coef_map
        assert abs(coef_map["x2"].coef - (-1.5)) < 0.5, f"x2 coef={coef_map['x2'].coef}"

    def test_fe_within_r_squared_positive(self) -> None:
        """Within R-squared should be positive for a well-specified FE model."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.within_r_squared is not None
        assert result.within_r_squared > 0, f"within_r2={result.within_r_squared}"

    def test_fe_standard_errors_positive(self) -> None:
        """All standard errors should be positive."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        for c in result.coefficients:
            assert c.se > 0, f"Non-positive SE for {c.name}: {c.se}"

    def test_fe_entity_count_and_time_count(self) -> None:
        """Entity count and time period count should match input data."""
        data = make_panel_data(n_entities=30, n_periods=8, seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.entity_count == 30
        assert result.time_count == 8

    def test_fe_t_statistics_present(self) -> None:
        """All coefficients should have valid t-statistics."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        for c in result.coefficients:
            assert not np.isnan(c.t_stat), f"NaN t-stat for {c.name}"
            assert abs(c.t_stat) > 0, f"Zero t-stat for {c.name}"

    def test_fe_model_type_and_panel_metadata(self) -> None:
        """FE result should have model_type='panel' and method='Panel FE'."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.model_type == "panel"
        assert result.method == "Panel FE"
        assert getattr(result, "panel_type", None) == "Panel FE"

    def test_fe_f_pooled_available(self) -> None:
        """FE model should provide F-test for poolability."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        fp = getattr(result, "f_pooled", None)
        assert fp is not None, "F-pooled should be available for FE"
        assert isinstance(fp, tuple) and len(fp) == 2
        assert fp[0] > 0, f"F-pooled stat should be > 0, got {fp[0]}"

    def test_fe_pvalues_in_range(self) -> None:
        """All p-values should be between 0 and 1."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        for c in result.coefficients:
            assert 0 <= c.pvalue <= 1, f"p-value out of range for {c.name}: {c.pvalue}"

    def test_fe_variable_labels_preserved(self) -> None:
        """Variable labels from build_variable_labels should be preserved."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", variable_labels=labels)

        assert "x1" in result.variable_labels
        assert "x2" in result.variable_labels

    def test_fe_ci_bounds_correct(self) -> None:
        """CI lower < coefficient < CI upper for all coefficients."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        for c in result.coefficients:
            assert c.ci_lower < c.coef < c.ci_upper, (
                f"CI bounds wrong for {c.name}: {c.ci_lower} < {c.coef} < {c.ci_upper}"
            )

    def test_fe_nobs_correct(self) -> None:
        """n_obs should match n_entities * n_periods."""
        data = make_panel_data(n_entities=30, n_periods=8, seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.n_obs == 30 * 8  # 240

    def test_fe_summary_output(self) -> None:
        """summary() should produce non-empty output."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        summary = result.summary()
        assert len(summary) > 0
        assert "Panel FE" in summary


# =========================================================================
# Tests: Random Effects
# =========================================================================


class TestRandomEffects:
    """Random Effects panel model tests."""

    def test_re_basic_fit(self) -> None:
        """RE should fit without error and return a valid result."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "random"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y", variable_labels=labels)

        assert isinstance(result, ModelResult)
        assert result.model_type == "panel"
        assert len(result.coefficients) >= 2
        assert result.n_obs > 0

    def test_re_coefficients_differ_from_fe(self) -> None:
        """RE coefficients should differ from FE (as is expected in panel data)."""
        data = make_panel_data(seed=42)
        spec_fe = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec_fe.entity_var = "entity"
        spec_fe.time_var = "time"
        spec_fe.panel_model = "fixed"

        spec_re = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec_re.entity_var = "entity"
        spec_re.time_var = "time"
        spec_re.panel_model = "random"

        fe_fitted, _ = run_panel(data, spec_fe)
        re_fitted, _ = run_panel(data, spec_re)
        fe_result = extract_panel(fe_fitted, dep_var="y")
        re_result = extract_panel(re_fitted, dep_var="y")

        fe_coefs = {c.name: c.coef for c in fe_result.coefficients}
        re_coefs = {c.name: c.coef for c in re_result.coefficients}

        # At least one common coefficient should differ
        common = set(fe_coefs.keys()) & set(re_coefs.keys())
        differences = [abs(fe_coefs[k] - re_coefs[k]) > 1e-10 for k in common]
        assert any(differences), (
            f"All coefficients identical between FE and RE: {fe_coefs} vs {re_coefs}"
        )

    def test_re_between_r_squared_present(self) -> None:
        """RE result should have between R-squared."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "random"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.between_r_squared is not None

    def test_re_overall_r_squared_present(self) -> None:
        """RE result should have overall R-squared."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "random"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.overall_r_squared is not None

    def test_re_method_is_panel_re(self) -> None:
        """RE result should have method='Panel RE'."""
        data = make_panel_data(seed=42)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "random"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.method == "Panel RE"
        assert result.model_type == "panel"

    def test_re_entity_and_time_counts(self) -> None:
        """RE should report correct entity and time counts."""
        data = make_panel_data(n_entities=20, n_periods=5, seed=123)
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "random"

        fitted, labels = run_panel(data, spec)
        result = extract_panel(fitted, dep_var="y")

        assert result.entity_count == 20
        assert result.time_count == 5


# =========================================================================
# Tests: Error handling
# =========================================================================


class TestPanelErrors:
    """Error-handling tests for the panel engine."""

    def test_missing_entity_var_raises(self) -> None:
        """run_panel should raise ValueError when entity_var is missing."""
        data = make_panel_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        # No entity_var set

        with pytest.raises(ValueError, match="entity_var"):
            run_panel(data, spec)

    def test_missing_time_var_raises(self) -> None:
        """run_panel should raise ValueError when time_var is missing."""
        data = make_panel_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        # No time_var set

        with pytest.raises(ValueError, match="time_var"):
            run_panel(data, spec)

    def test_single_entity_raises(self) -> None:
        """run_panel should raise ValueError with a single entity."""
        # Create data with 1 entity, 10 periods
        rng = np.random.default_rng(42)
        n = 10
        df = pd.DataFrame({
            "entity": [0] * n,
            "time": np.arange(n),
            "y": rng.normal(0, 1, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        })
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        with pytest.raises(ValueError, match="at least 2 entities"):
            run_panel(df, spec)

    def test_bad_panel_model_raises(self) -> None:
        """run_panel should raise ValueError for unknown panel_model."""
        data = make_panel_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "time"
        spec.panel_model = "between"  # Not supported

        with pytest.raises(ValueError, match="Unknown panel model"):
            run_panel(data, spec)

    def test_entity_var_not_in_columns_raises(self) -> None:
        """run_panel should raise ValueError when entity column doesn't exist."""
        data = make_panel_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "nonexistent"
        spec.time_var = "time"
        spec.panel_model = "fixed"

        with pytest.raises(ValueError, match="not found in data"):
            run_panel(data, spec)

    def test_time_var_not_in_columns_raises(self) -> None:
        """run_panel should raise ValueError when time column doesn't exist."""
        data = make_panel_data()
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
        spec.entity_var = "entity"
        spec.time_var = "nonexistent"
        spec.panel_model = "fixed"

        with pytest.raises(ValueError, match="not found in data"):
            run_panel(data, spec)

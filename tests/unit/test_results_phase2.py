# encoding: utf-8
"""Phase 2 unit tests for enhanced results functionality.

Tests cover:
    - ModelResult.to_summary_dict()
    - ModelResult.anova_table()
    - ModelResult.to_latex_row()
    - compare_models()
    - summary_generator functions
    - Enhanced statistics (anova_oneway, freq_table)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec, build_formula
from src.results.statistics import anova_oneway, freq_table
from src.results.summary_generator import (
    generate_assumption_check_text,
    generate_coefficient_interpretation,
    generate_summary_text,
)
from src.results.table import (
    CoefficientRow,
    ModelResult,
    _pvalue_label,
    _significance_stars,
    compare_models,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_ols.csv"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Load the sample OLS test dataset."""
    return pd.read_csv(SAMPLE_CSV, encoding="utf-8")


@pytest.fixture
def fitted_result(sample_data: pd.DataFrame) -> ModelResult:
    """Fit a simple OLS model and return the ModelResult."""
    spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
    fitter = ModelFitter()
    return fitter.fit(spec, sample_data)


@pytest.fixture
def fitted_result_with_cat(sample_data: pd.DataFrame) -> ModelResult:
    """Fit OLS with categorical variables included."""
    spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"])
    fitter = ModelFitter()
    return fitter.fit(spec, sample_data)


# =========================================================================
# Test: ModelResult.to_summary_dict()
# =========================================================================
class TestModelResultToSummaryDict:
    """Verify the summary dictionary contains all expected fields."""

    def test_to_summary_dict_keys(self, fitted_result: ModelResult) -> None:
        """Check that all expected keys are present."""
        d = fitted_result.to_summary_dict()

        expected_keys = [
            "dep_var",
            "n_obs",
            "n_params",
            "df_resid",
            "r_squared",
            "adj_r_squared",
            "rmse",
            "log_likelihood",
            "aic",
            "bic",
            "method",
            "specification",
            "f_statistic",
            "f_pvalue",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

        # Verify types
        assert d["dep_var"] == "y"
        assert isinstance(d["n_obs"], int)
        assert d["n_obs"] == 200
        assert isinstance(d["n_params"], int)
        assert d["n_params"] == 3
        assert isinstance(d["r_squared"], float)
        assert d["r_squared"] > 0
        assert isinstance(d["f_statistic"], float)
        assert d["f_statistic"] > 0
        assert isinstance(d["f_pvalue"], float)
        assert d["f_pvalue"] < 0.05

    def test_to_summary_dict_no_f_stat(self) -> None:
        """Verify behavior when f_statistic is None."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.0, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=100,
            n_params=1,
            df_resid=99,
        )
        d = result.to_summary_dict()
        assert d["f_statistic"] is None
        assert d["f_pvalue"] is None


# =========================================================================
# Test: ModelResult.anova_table()
# =========================================================================
class TestAnovaTable:
    """Verify the ANOVA table structure and values."""

    def test_anova_table_structure(self, fitted_result: ModelResult) -> None:
        """Check columns and rows of the ANOVA table."""
        anova = fitted_result.anova_table()

        # Expected columns
        expected_cols = ["来源", "SS", "df", "MS", "F", "p-value"]
        for col in expected_cols:
            assert col in anova.columns, f"Missing column: {col}"

        # Expected rows
        assert len(anova) == 3
        sources = anova["来源"].tolist()
        assert "回归(Explained)" in sources
        assert "残差(Residual)" in sources
        assert "总计(Total)" in sources

    def test_anova_table_ss_decomposition(self, fitted_result: ModelResult) -> None:
        """Verify that SS_explained + SS_residual ≈ SS_total."""
        anova = fitted_result.anova_table()
        ss_explained = anova.loc[anova["来源"] == "回归(Explained)", "SS"].values[0]
        ss_residual = anova.loc[anova["来源"] == "残差(Residual)", "SS"].values[0]
        ss_total = anova.loc[anova["来源"] == "总计(Total)", "SS"].values[0]

        # Allow for floating-point rounding
        assert abs((ss_explained + ss_residual) - ss_total) < 0.01, (
            f"SS sum mismatch: {ss_explained} + {ss_residual} != {ss_total}"
        )

    def test_anova_table_df(self, fitted_result: ModelResult) -> None:
        """Verify degrees of freedom."""
        anova = fitted_result.anova_table()
        df_explained = anova.loc[anova["来源"] == "回归(Explained)", "df"].values[0]
        df_residual = anova.loc[anova["来源"] == "残差(Residual)", "df"].values[0]
        df_total = anova.loc[anova["来源"] == "总计(Total)", "df"].values[0]

        assert df_explained == 2  # n_params - 1
        assert df_residual == 197  # n_obs - n_params
        assert df_total == 199  # n_obs - 1
        assert df_explained + df_residual == df_total

    def test_anova_table_f_stat(self, fitted_result: ModelResult) -> None:
        """Verify F-statistic in ANOVA matches model F-statistic."""
        anova = fitted_result.anova_table()
        anova_f = anova.loc[anova["来源"] == "回归(Explained)", "F"].values[0]

        model_f = fitted_result.f_statistic[0] if fitted_result.f_statistic else None
        assert model_f is not None
        assert abs(anova_f - model_f) < 0.001, (
            f"ANOVA F ({anova_f}) != model F ({model_f})"
        )

    def test_anova_table_single_coefficient_result(self) -> None:
        """ANOVA with only one coefficient (no df_explained)."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="y", coef=0.0, se=0.1, t_stat=0.0,
                    pvalue=1.0, ci_lower=-0.2, ci_upper=0.2
                )
            ],
            n_obs=10,
            n_params=1,
            df_resid=9,
            r_squared=0.0,
        )
        anova = result.anova_table()
        assert anova is not None
        assert len(anova) == 3


# =========================================================================
# Test: ModelResult.to_latex_row()
# =========================================================================
class TestModelResultToLatexRow:
    """Verify LaTeX row generation."""

    def test_to_latex_row_format(self, fitted_result: ModelResult) -> None:
        """Check that LaTeX row ends with \\\\ and contains expected fields."""
        latex = fitted_result.to_latex_row()
        assert latex.endswith("\\\\")
        assert fitted_result.dep_var in latex
        assert str(fitted_result.n_obs) in latex
        assert "&" in latex  # tab separator

    def test_to_latex_row_components(self, fitted_result: ModelResult) -> None:
        """Verify all major components appear in the LaTeX row."""
        latex = fitted_result.to_latex_row()
        parts = latex.split(" & ")
        # dep_var, n, r2, adj_r2, f, fp, aic, bic
        assert len(parts) == 8


# =========================================================================
# Test: compare_models()
# =========================================================================
class TestCompareModels:
    """Verify multi-model comparison table."""

    def test_compare_models_empty(self) -> None:
        """Empty input should produce empty DataFrame."""
        result = compare_models([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_compare_models_two_models(self, sample_data: pd.DataFrame) -> None:
        """Compare two models with different specs."""
        spec1 = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        spec2 = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x4"])
        fitter = ModelFitter()
        result1 = fitter.fit(spec1, sample_data)
        result2 = fitter.fit(spec2, sample_data)

        comparison = compare_models([result1, result2])

        # Should have coefficient rows + 5 stat rows
        assert len(comparison) >= 5

        # Should include summary statistics
        stat_labels = comparison["变量"].tolist()
        for label in ["N", "R²", "Adj-R²", "AIC", "BIC"]:
            assert label in stat_labels, f"Missing stat: {label}"

        # Coefficient rows should show coef (se)
        x1_rows = comparison[comparison["变量"] == "x1"]
        if len(x1_rows) > 0:
            x1_cell = x1_rows.iloc[0, 1]
            assert isinstance(x1_cell, str)
            assert "(" in x1_cell
            assert ")" in x1_cell

    def test_compare_models_single(self, sample_data: pd.DataFrame) -> None:
        """Single model comparison should still produce valid table."""
        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)

        comparison = compare_models([result])
        assert "Model 1" in comparison.columns or comparison.shape[1] > 1
        assert len(comparison) >= 5  # coef rows + stat rows


# =========================================================================
# Test: summary_generator
# =========================================================================
class TestSummaryGenerator:
    """Verify text summary generation."""

    def test_generate_summary_text_structure(self, fitted_result: ModelResult) -> None:
        """Check that summary text contains expected keywords."""
        text = generate_summary_text(fitted_result)

        assert "OLS" in text or "回归" in text
        assert fitted_result.dep_var in text
        assert "R²" in text
        assert "AIC" in text
        assert "BIC" in text
        assert len(text) > 50  # Should be substantial

    def test_generate_summary_text_no_f_stat(self) -> None:
        """Summary still works when f_statistic is None."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.0, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=50,
            n_params=1,
            df_resid=49,
            r_squared=0.5,
            aic=100.0,
            bic=105.0,
            rmse=0.5,
        )
        text = generate_summary_text(result)
        assert "R²" in text
        assert "AIC" in text

    def test_generate_coefficient_interpretation_positive(self) -> None:
        """Positive coefficient interpretation."""
        row = CoefficientRow(
            name="x1", coef=2.5, se=0.5, t_stat=5.0,
            pvalue=0.001, ci_lower=1.5, ci_upper=3.5
        )
        text = generate_coefficient_interpretation(row, "y")
        assert "增加" in text
        assert "2.5000" in text
        assert "x1" in text
        assert "y" in text

    def test_generate_coefficient_interpretation_negative(self) -> None:
        """Negative coefficient interpretation should say '减少'."""
        row = CoefficientRow(
            name="x1", coef=-1.5, se=0.3, t_stat=-5.0,
            pvalue=0.01, ci_lower=-2.1, ci_upper=-0.9
        )
        text = generate_coefficient_interpretation(row, "y")
        assert "减少" in text
        assert "1.5000" in text

    def test_generate_assumption_check_text_basic(self, fitted_result: ModelResult) -> None:
        """Assumption check text should contain section headers."""
        text = generate_assumption_check_text(fitted_result)
        assert "假设检验检查" in text
        assert "多重共线性" in text
        assert "残差" in text

    def test_generate_assumption_check_text_with_vif(
        self, sample_data: pd.DataFrame, fitted_result: ModelResult
    ) -> None:
        """Assumption check text should include VIF results."""
        from src.modeling.diagnostics import vif

        spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x4"])
        fitter = ModelFitter()
        result = fitter.fit(spec, sample_data)
        vif_df = vif(sample_data, spec)
        text = generate_assumption_check_text(result, vif_df=vif_df)
        assert "VIF" in text or "多重共线性" in text

    def test_generate_assumption_check_text_with_residual_tests(
        self, fitted_result: ModelResult
    ) -> None:
        """Assumption check text should include residual diagnostics."""
        from src.modeling.diagnostics import residual_tests

        residuals = np.random.default_rng(42).normal(0, 1, 100)
        rt = residual_tests(residuals)
        text = generate_assumption_check_text(
            fitted_result, residual_tests=rt
        )
        assert "Shapiro-Wilk" in text or "Durbin-Watson" in text
        assert "残差" in text


# =========================================================================
# Test: anova_oneway
# =========================================================================
class TestAnovaOneway:
    """Verify one-way ANOVA function."""

    def test_anova_oneway_basic(self, sample_data: pd.DataFrame) -> None:
        """Basic ANOVA with categorical group variable."""
        result = anova_oneway(sample_data, dv="y", group="x3")

        expected_keys = [
            "f_statistic", "p_value", "df_between", "df_within",
            "ss_between", "ss_within", "ss_total",
            "group_means", "group_counts", "group_stds",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        assert isinstance(result["f_statistic"], float)
        assert isinstance(result["p_value"], float)
        assert result["df_between"] >= 1
        assert result["df_within"] >= 1
        assert result["ss_total"] > 0
        assert len(result["group_means"]) >= 2
        assert len(result["group_counts"]) >= 2

    def test_anova_oneway_invalid_column(self, sample_data: pd.DataFrame) -> None:
        """Should raise ValueError for missing columns."""
        with pytest.raises(ValueError, match="not found in data"):
            anova_oneway(sample_data, dv="nonexistent", group="x3")

        with pytest.raises(ValueError, match="not found in data"):
            anova_oneway(sample_data, dv="y", group="nonexistent")

    def test_anova_oneway_insufficient_data(self) -> None:
        """Should raise ValueError when too few observations."""
        data = pd.DataFrame({"y": [1.0, 2.0], "g": ["A", "B"]})
        with pytest.raises(ValueError, match="least 3"):
            anova_oneway(data, dv="y", group="g")


# =========================================================================
# Test: freq_table
# =========================================================================
class TestFreqTable:
    """Verify frequency table function."""

    def test_freq_table_basic(self, sample_data: pd.DataFrame) -> None:
        """Basic frequency table for categorical variable."""
        ft = freq_table(sample_data, col="x3")

        expected_cols = ["类别", "频数", "百分比(%)", "累积百分比(%)"]
        for col in expected_cols:
            assert col in ft.columns, f"Missing column: {col}"

        # Should have 3 categories (A, B, C)
        assert len(ft) == 3

        # Frequencies should sum to total observations
        total_freq = ft["频数"].sum()
        assert total_freq == len(sample_data)

        # Percentages should sum to ~100
        assert abs(ft["百分比(%)"].sum() - 100.0) < 0.1

        # Cumulative percentage should be monotonic
        cum_pcts = ft["累积百分比(%)"].values
        assert all(cum_pcts[i] <= cum_pcts[i + 1] for i in range(len(cum_pcts) - 1))

        # Last cumulative percentage should be exactly 100
        assert cum_pcts[-1] == 100.0

    def test_freq_table_invalid_column(self, sample_data: pd.DataFrame) -> None:
        """Should raise ValueError for missing column."""
        with pytest.raises(ValueError, match="not found in data"):
            freq_table(sample_data, col="nonexistent")

    def test_freq_table_numeric(self) -> None:
        """Frequency table should work for numeric columns too."""
        data = pd.DataFrame({"x": [1, 1, 2, 2, 2, 3]})
        ft = freq_table(data, col="x")
        assert len(ft) == 3
        # Most frequent value should be 2
        assert ft.loc[ft["频数"].idxmax(), "类别"] == "2"

    def test_freq_table_with_missing(self) -> None:
        """Frequency table should ignore NaN values."""
        data = pd.DataFrame({"x": ["A", "A", "B", None, "C", None]})
        ft = freq_table(data, col="x")
        # Should ignore 2 NaN values, so total freq = 4
        assert ft["频数"].sum() == 4
        assert len(ft) == 3


# =========================================================================
# Test: _pvalue_label helper
# =========================================================================
class TestPvalueLabel:
    """Verify the _pvalue_label helper."""

    def test_pvalue_label_thresholds(self) -> None:
        assert _pvalue_label(0.005) == "p<0.01"
        assert _pvalue_label(0.03) == "p<0.05"
        assert _pvalue_label(0.07) == "p<0.1"
        assert _pvalue_label(0.5) == "p>=0.1"
        # Strict less-than boundaries
        assert _pvalue_label(0.009) == "p<0.01"
        assert _pvalue_label(0.049) == "p<0.05"
        assert _pvalue_label(0.099) == "p<0.1"

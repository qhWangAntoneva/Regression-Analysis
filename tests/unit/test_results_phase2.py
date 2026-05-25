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


# =========================================================================
# Test: summary_generator — full branch coverage
# =========================================================================

class TestSummaryGeneratorBranches:
    """Achieve full branch coverage for generate_summary_text."""

    def test_f_p_less_than_001(self) -> None:
        """p < 0.001 branch: should say '<0.001'."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.5, se=0.2, t_stat=7.5,
                    pvalue=0.001, ci_lower=1.1, ci_upper=1.9
                )
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.65,
            adj_r_squared=0.64,
            aic=120.0,
            bic=125.0,
            rmse=0.8,
            f_statistic=(25.0, 0.0005),  # p < 0.001
            dep_var="y",
        )
        text = generate_summary_text(result)
        assert "p<0.001" in text

    def test_f_p_between_001_and_005(self) -> None:
        """p between 0.001 and 0.05: should show formatted p-value."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=0.8, se=0.3, t_stat=2.67,
                    pvalue=0.01, ci_lower=0.2, ci_upper=1.4
                )
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.20,
            adj_r_squared=0.19,
            aic=200.0,
            bic=205.0,
            rmse=0.9,
            f_statistic=(7.12, 0.0089),  # 0.001 < p < 0.05
            dep_var="y",
        )
        text = generate_summary_text(result)
        assert "F(" in text
        assert "p=0.0089" in text

    def test_f_p_borderline_significant(self) -> None:
        """p between 0.05 and 0.1: should say '边缘显著'."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=0.5, se=0.3, t_stat=1.67,
                    pvalue=0.1, ci_lower=-0.1, ci_upper=1.1
                )
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.03,
            adj_r_squared=0.02,
            aic=250.0,
            bic=255.0,
            rmse=1.0,
            f_statistic=(2.78, 0.072),  # 0.05 < p < 0.1
            dep_var="y",
        )
        text = generate_summary_text(result)
        assert "边缘显著" in text

    def test_f_p_not_significant(self) -> None:
        """p >= 0.1: should say '不显著'."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=0.1, se=0.2, t_stat=0.5,
                    pvalue=0.62, ci_lower=-0.3, ci_upper=0.5
                )
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.002,
            adj_r_squared=0.0,
            aic=300.0,
            bic=305.0,
            rmse=1.2,
            f_statistic=(0.25, 0.62),  # p >= 0.1
            dep_var="y",
        )
        text = generate_summary_text(result)
        assert "不显著" in text

    def test_no_r_squared(self) -> None:
        """r_squared is None: R-squared text should be omitted."""
        result = ModelResult(
            model_type="WLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.001, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=50,
            n_params=2,
            df_resid=48,
            r_squared=None,
            adj_r_squared=None,
            aic=100.0,
            bic=105.0,
            rmse=0.5,
        )
        text = generate_summary_text(result)
        # Should not contain "R²="
        assert "R²=" not in text

    def test_no_n_obs(self) -> None:
        """n_obs is None/0: sample size line should be omitted."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.001, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=0,
            n_params=2,
            df_resid=0,
            r_squared=0.5,
            aic=100.0,
            bic=105.0,
            rmse=0.5,
        )
        text = generate_summary_text(result)
        # N=0 should not produce N text (falsy)
        assert "N=0" not in text

    def test_no_dep_var_no_spec(self) -> None:
        """Empty dep_var and specification: defaults should be used."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.001, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=50,
            n_params=2,
            df_resid=48,
            r_squared=0.5,
            aic=100.0,
            bic=105.0,
            rmse=0.5,
            dep_var="",
            specification="",
        )
        text = generate_summary_text(result)
        assert "因变量" in text
        assert "未指定" in text

    def test_adj_r_squared_present(self) -> None:
        """When adj_r_squared is available, it should appear."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.001, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.75,
            adj_r_squared=0.74,
            aic=100.0,
            bic=105.0,
            rmse=0.5,
        )
        text = generate_summary_text(result)
        assert "调整R²" in text
        assert "0.7400" in text

    def test_adj_r_squared_none(self) -> None:
        """When adj_r_squared is None, only R² should appear."""
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.1, t_stat=10.0,
                    pvalue=0.001, ci_lower=0.8, ci_upper=1.2
                )
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.75,
            adj_r_squared=None,
            aic=100.0,
            bic=105.0,
            rmse=0.5,
        )
        text = generate_summary_text(result)
        assert "调整R²" not in text


# =========================================================================
# Test: generate_coefficient_interpretation — all branches
# =========================================================================

class TestCoefficientInterpretationBranches:
    """Cover all significance and sign branches in coefficient interpretation."""

    def test_highly_significant_0001(self) -> None:
        """p < 0.001: should show 'p<0.001'."""
        row = CoefficientRow(
            name="edu", coef=3.5, se=0.5, t_stat=7.0,
            pvalue=0.0005, ci_lower=2.5, ci_upper=4.5
        )
        text = generate_coefficient_interpretation(row, "wage")
        assert "p<0.001" in text

    def test_significant_001(self) -> None:
        """p < 0.01: should show 'p<0.01' (from _sig_star_text)."""
        row = CoefficientRow(
            name="exp", coef=0.8, se=0.2, t_stat=4.0,
            pvalue=0.005, ci_lower=0.4, ci_upper=1.2
        )
        text = generate_coefficient_interpretation(row, "wage")
        assert "p<0.01" in text

    def test_significant_005(self) -> None:
        """p < 0.05: should show 'p<0.05'."""
        row = CoefficientRow(
            name="tenure", coef=0.5, se=0.2, t_stat=2.5,
            pvalue=0.03, ci_lower=0.1, ci_upper=0.9
        )
        text = generate_coefficient_interpretation(row, "wage")
        assert "p<0.05" in text

    def test_weakly_significant(self) -> None:
        """p < 0.1: should show 'p<0.1'."""
        row = CoefficientRow(
            name="region", coef=0.2, se=0.12, t_stat=1.67,
            pvalue=0.08, ci_lower=-0.04, ci_upper=0.44
        )
        text = generate_coefficient_interpretation(row, "income")
        assert "p<0.1" in text

    def test_not_significant(self) -> None:
        """p >= 0.1: should show 'p>=0.1（不显著）'."""
        row = CoefficientRow(
            name="noise", coef=0.05, se=0.1, t_stat=0.5,
            pvalue=0.62, ci_lower=-0.15, ci_upper=0.25
        )
        text = generate_coefficient_interpretation(row, "outcome")
        assert "p>=0.1" in text
        assert "不显著" in text

    def test_zero_coefficient(self) -> None:
        """Zero coefficient: direction should be '增加' (>=0), value 0.0."""
        row = CoefficientRow(
            name="zero_var", coef=0.0, se=0.1, t_stat=0.0,
            pvalue=1.0, ci_lower=-0.2, ci_upper=0.2
        )
        text = generate_coefficient_interpretation(row, "y")
        # coef >= 0 so direction should be 增加
        assert "增加" in text
        assert "0.0000" in text

    def test_very_small_positive_coefficient(self) -> None:
        """Very small positive coefficient should use '增加'."""
        row = CoefficientRow(
            name="tiny", coef=0.0001, se=0.0001, t_stat=1.0,
            pvalue=0.32, ci_lower=-0.0001, ci_upper=0.0003
        )
        text = generate_coefficient_interpretation(row, "y")
        assert "增加" in text
        assert "0.0001" in text

    def test_very_small_negative_coefficient(self) -> None:
        """Very small negative coefficient should use '减少'."""
        row = CoefficientRow(
            name="tiny_neg", coef=-0.0001, se=0.0001, t_stat=-1.0,
            pvalue=0.32, ci_lower=-0.0003, ci_upper=0.0001
        )
        text = generate_coefficient_interpretation(row, "y")
        assert "减少" in text
        assert "0.0001" in text

    def test_large_coefficient(self) -> None:
        """Large coefficient value."""
        row = CoefficientRow(
            name="big", coef=12345.6789, se=500.0, t_stat=24.69,
            pvalue=0.0001, ci_lower=11345.0, ci_upper=13346.0
        )
        text = generate_coefficient_interpretation(row, "y")
        assert "增加" in text
        assert "12345.6789" in text

    def test_contains_keywords_in_order(self) -> None:
        """Verify the Chinese interpretation template is well-formed."""
        row = CoefficientRow(
            name="x1", coef=1.5, se=0.3, t_stat=5.0,
            pvalue=0.002, ci_lower=0.9, ci_upper=2.1
        )
        text = generate_coefficient_interpretation(row, "response")
        assert "在其他变量保持不变的情况下" in text
        assert "每增加一个单位" in text
        assert "平均" in text
        assert "个单位" in text


# =========================================================================
# Test: generate_assumption_check_text — full branch coverage
# =========================================================================

class TestAssumptionCheckTextBranches:
    """Cover every branch in generate_assumption_check_text()."""

    # --- VIF tests ---

    def test_high_vif(self, fitted_result: ModelResult) -> None:
        """VIF with 'High' diagnosis should report severe multicollinearity."""
        import pandas as pd

        vif_df = pd.DataFrame({
            "variable": ["const", "x1", "x2"],
            "vif": [1.0, 25.0, 30.0],
            "diagnosis": ["Low", "High", "High"],
        })
        text = generate_assumption_check_text(fitted_result, vif_df=vif_df)
        assert "严重多重共线性" in text
        assert "x1(VIF=25.00)" in text
        assert "x2(VIF=30.00)" in text

    def test_moderate_vif(self, fitted_result: ModelResult) -> None:
        """VIF with 'Moderate' diagnosis (no 'High') should report moderate."""
        import pandas as pd

        vif_df = pd.DataFrame({
            "variable": ["const", "x1", "x2"],
            "vif": [1.0, 5.5, 6.2],
            "diagnosis": ["Low", "Moderate", "Moderate"],
        })
        text = generate_assumption_check_text(fitted_result, vif_df=vif_df)
        assert "中等程度多重共线性" in text
        assert "x1(VIF=5.50)" in text
        assert "x2(VIF=6.20)" in text

    def test_low_vif_all_ok(self, fitted_result: ModelResult) -> None:
        """All VIF < 5: should report no multicollinearity."""
        import pandas as pd

        vif_df = pd.DataFrame({
            "variable": ["const", "x1", "x2", "x3"],
            "vif": [1.0, 1.2, 2.3, 3.4],
            "diagnosis": ["Low", "Low", "Low", "Low"],
        })
        text = generate_assumption_check_text(fitted_result, vif_df=vif_df)
        assert "未发现严重的多重共线性" in text

    def test_vif_empty_dataframe(self, fitted_result: ModelResult) -> None:
        """Empty VIF DataFrame (not None, but .empty) should show no VIF data."""
        import pandas as pd

        vif_df = pd.DataFrame(columns=["variable", "vif", "diagnosis"])
        text = generate_assumption_check_text(fitted_result, vif_df=vif_df)
        assert "未提供VIF数据" in text

    def test_vif_with_const_ignored(self, fitted_result: ModelResult) -> None:
        """VIF with only const (which is ignored) should be treated as no VIF."""
        import pandas as pd

        vif_df = pd.DataFrame({
            "variable": ["const"],
            "vif": [100.0],
            "diagnosis": ["High"],
        })
        text = generate_assumption_check_text(fitted_result, vif_df=vif_df)
        # const is ignored, no other vars → no multicollinearity
        assert "未发现严重的多重共线性" in text

    def test_vif_with_intercept_ignored(self, fitted_result: ModelResult) -> None:
        """VIF with 'Intercept' (which is ignored) should be treated as no issue."""
        import pandas as pd

        vif_df = pd.DataFrame({
            "variable": ["Intercept", "x1"],
            "vif": [50.0, 1.5],
            "diagnosis": ["High", "Low"],
        })
        text = generate_assumption_check_text(fitted_result, vif_df=vif_df)
        # Intercept is ignored, x1 is Low
        assert "未发现严重的多重共线性" in text

    # --- Residual diagnostic tests ---

    def test_shapiro_normal_yes(self, fitted_result: ModelResult) -> None:
        """Shapiro-Wilk normal: should report approximate normality."""
        rt = {"shapiro_normal": "Yes"}
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "近似正态分布" in text

    def test_shapiro_normal_no_with_pvalue(self, fitted_result: ModelResult) -> None:
        """Shapiro-Wilk not normal with p-value: should report non-normality."""
        rt = {"shapiro_normal": "No", "shapiro_pvalue": 0.001}
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "不服从正态分布" in text
        assert "p=0.0010" in text

    def test_shapiro_normal_no_without_pvalue(self, fitted_result: ModelResult) -> None:
        """Shapiro-Wilk not normal without p-value: should still report issue."""
        rt = {"shapiro_normal": "No"}
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "不服从正态分布" in text

    def test_shapiro_unknown_value(self, fitted_result: ModelResult) -> None:
        """Unexpected shapiro_normal value: should be directly displayed."""
        rt = {"shapiro_normal": "Insufficient data"}
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "残差正态性检查: Insufficient data。" in text

    def test_dw_no_autocorrelation(self, fitted_result: ModelResult) -> None:
        """DW indicates no autocorrelation."""
        rt = {
            "shapiro_normal": "Yes",
            "dw_autocorrelation": "None",
            "dw_stat": 2.05,
        }
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "无明显自相关" in text
        assert "2.0500" in text

    def test_dw_positive_autocorrelation(self, fitted_result: ModelResult) -> None:
        """DW indicates positive autocorrelation."""
        rt = {
            "shapiro_normal": "Yes",
            "dw_autocorrelation": "Positive",
            "dw_stat": 0.85,
        }
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "存在Positive自相关" in text
        assert "标准误可能被低估" in text

    def test_dw_negative_autocorrelation(self, fitted_result: ModelResult) -> None:
        """DW indicates negative autocorrelation."""
        rt = {
            "shapiro_normal": "Yes",
            "dw_autocorrelation": "Negative",
            "dw_stat": 3.2,
        }
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "存在Negative自相关" in text

    def test_dw_insufficient_data(self, fitted_result: ModelResult) -> None:
        """DW with 'Insufficient data' status."""
        rt = {
            "dw_autocorrelation": "Insufficient data",
            "dw_stat": float("nan"),
        }
        text = generate_assumption_check_text(fitted_result, residual_tests=rt)
        assert "数据不足以判断" in text

    def test_combined_vif_and_residual(self, fitted_result: ModelResult) -> None:
        """Full combination: VIF + normality + DW."""
        import pandas as pd

        vif_df = pd.DataFrame({
            "variable": ["const", "x1", "x2"],
            "vif": [1.0, 1.3, 2.1],
            "diagnosis": ["Low", "Low", "Low"],
        })
        rt = {
            "shapiro_normal": "Yes",
            "dw_autocorrelation": "None",
            "dw_stat": 1.98,
        }
        text = generate_assumption_check_text(
            fitted_result, vif_df=vif_df, residual_tests=rt
        )
        assert "未发现严重的多重共线性" in text
        assert "近似正态分布" in text
        assert "无明显自相关" in text


# =========================================================================
# Test: ModelResult with model_type="logit" (Phase 5.1)
# =========================================================================
class TestModelResultLogit:
    """ModelResult behaviour when model_type is 'logit'."""

    @pytest.fixture
    def logit_result(self) -> ModelResult:
        """Build a representative logit ModelResult."""
        return ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="Intercept", coef=-0.5, se=0.3, t_stat=-1.67,
                    pvalue=0.095, ci_lower=-1.1, ci_upper=0.1
                ),
                CoefficientRow(
                    name="x1", coef=1.2, se=0.4, t_stat=3.0,
                    pvalue=0.003, ci_lower=0.4, ci_upper=2.0
                ),
            ],
            n_obs=200,
            n_params=2,
            df_resid=198,
            rmse=None,
            pseudo_r_squared=0.15,
            log_likelihood=-120.0,
            aic=244.0,
            bic=250.0,
            llr=25.0,
            llr_pvalue=0.0001,
            dep_var="y_bin",
            method="Logit",
        )

    def test_logit_summary_dict(self, logit_result: ModelResult) -> None:
        """to_summary_dict() for logit should have logit-specific fields."""
        d = logit_result.to_summary_dict()

        assert d["model_type"] == "logit"
        assert d["pseudo_r_squared"] == 0.15
        assert d["r_squared"] is None
        assert d["rmse"] is None
        assert d["f_statistic"] is None
        assert d["f_pvalue"] is None
        assert d["llr"] == 25.0
        assert d["llr_pvalue"] == 0.0001

    def test_logit_to_latex_row(self, logit_result: ModelResult) -> None:
        """Logit to_latex_row() uses pseudo R² and LR test."""
        latex = logit_result.to_latex_row()
        assert latex.endswith("\\\\")
        parts = latex.split(" & ")
        assert len(parts) == 7  # dep_var, n, pseudo_r2, llr, llr_p, aic, bic
        assert "y_bin" in latex
        assert "200" in latex

    def test_logit_anova_empty(self, logit_result: ModelResult) -> None:
        """anova_table() returns empty DataFrame for logit."""
        anova = logit_result.anova_table()
        assert anova.empty

    def test_logit_to_dataframe_uses_z(self, logit_result: ModelResult) -> None:
        """Logit to_dataframe() should have 'z值' column."""
        df = logit_result.to_dataframe()
        assert "z值" in df.columns
        assert "t值" not in df.columns

    def test_significance_stars_for_logit_pvalues(self) -> None:
        """_significance_stars works correctly with typical logit p-values."""
        assert _significance_stars(0.001) == "***"
        assert _significance_stars(0.03) == "**"
        assert _significance_stars(0.07) == "*"
        assert _significance_stars(0.5) == ""

    def test_logit_to_dataframe_has_or(self, logit_result: ModelResult) -> None:
        """Logit to_dataframe() should include OR(exp(B)) column."""
        df = logit_result.to_dataframe()
        assert "OR(exp(B))" in df.columns
        import math
        # Check first row: Intercept coef=-0.5, OR = exp(-0.5)
        or_val = df["OR(exp(B))"].iloc[0]
        assert abs(or_val - math.exp(-0.5)) < 0.001

    def test_ols_to_dataframe_no_or(self, fitted_result: ModelResult) -> None:
        """OLS to_dataframe() should NOT have OR column."""
        df = fitted_result.to_dataframe()
        assert "OR(exp(B))" not in df.columns


# =========================================================================
# Test: LatexRenderer with logit models (TODO 5.1.6)
# =========================================================================
class TestLatexRendererLogit:
    """LaTeX renderer behaviour for logit models."""

    @pytest.fixture
    def logit_result(self) -> ModelResult:
        """Build a representative logit ModelResult."""
        return ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="Intercept", coef=-0.5, se=0.3, t_stat=-1.67,
                    pvalue=0.095, ci_lower=-1.1, ci_upper=0.1
                ),
                CoefficientRow(
                    name="x1", coef=1.2, se=0.4, t_stat=3.0,
                    pvalue=0.003, ci_lower=0.4, ci_upper=2.0
                ),
            ],
            n_obs=200,
            n_params=2,
            df_resid=198,
            rmse=None,
            pseudo_r_squared=0.15,
            log_likelihood=-120.0,
            aic=244.0,
            bic=250.0,
            llr=25.0,
            llr_pvalue=0.0001,
            dep_var="y_bin",
            method="Logit",
        )

    def test_render_single_logit_uses_or_and_z(self, logit_result: ModelResult) -> None:
        """render_single for logit should use OR column and z-statistic."""
        from src.export.latex_renderer import LatexRenderer

        latex = LatexRenderer.render_single(logit_result)
        # Should use z instead of t
        assert "$z$" in latex
        assert "$t$" not in latex
        # Should show OR (exp(B))
        assert "OR (exp($B$))" in latex
        # Should show pseudo R-squared
        assert "pseudo-$R^2$" in latex
        # Should show LR chi2
        assert "LR $\\chi^2$" in latex
        # Should NOT show R-squared or F-statistic
        assert "R$^2$" not in latex
        assert "F-statistic" not in latex

    def test_render_single_logit_odds_ratios(self, logit_result: ModelResult) -> None:
        """Logit render_single should show exponentiated coefficients."""
        from src.export.latex_renderer import LatexRenderer
        import math

        latex = LatexRenderer.render_single(logit_result)
        # OR for Intercept: exp(-0.5) ≈ 0.6065
        or_intercept = math.exp(-0.5)
        # OR for x1: exp(1.2) ≈ 3.3201
        or_x1 = math.exp(1.2)
        # Check that OR values appear (formatted to 4 decimal places)
        assert f"{or_intercept:.4f}" in latex
        assert f"{or_x1:.4f}" in latex

    def test_render_single_logit_caption(self, logit_result: ModelResult) -> None:
        """render_single with default caption should use 'Logistic Regression'."""
        from src.export.latex_renderer import LatexRenderer

        latex = LatexRenderer.render_single(
            logit_result, title="Table 1"
        )
        assert "Logistic Regression Results" in latex
        assert "OLS" not in latex

    def test_render_comparison_mixed_models(self, fitted_result: ModelResult) -> None:
        """render_comparison with mixed OLS+logit should add model type row."""
        from src.export.latex_renderer import LatexRenderer

        logit = ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="Intercept", coef=-0.5, se=0.3, t_stat=-1.67,
                    pvalue=0.095, ci_lower=-1.1, ci_upper=0.1
                ),
            ],
            n_obs=200,
            n_params=1,
            df_resid=199,
            pseudo_r_squared=0.15,
            aic=244.0,
            bic=250.0,
            dep_var="y_bin",
        )

        latex = LatexRenderer.render_comparison(
            [fitted_result, logit],
            model_labels=["OLS Model", "Logit Model"],
        )
        # Should have a model type row
        assert "Model type" in latex
        assert "Logit" in latex
        # Should show both R² and Pseudo-R² rows
        assert "Pseudo-$R^2$" in latex
        # Should show both F-statistic and LR chi2 rows
        assert "F-statistic" in latex
        assert "LR $\\chi^2$" in latex

    def test_render_comparison_all_logit(self) -> None:
        """render_comparison with all-logit should show model type and logit stats."""
        from src.export.latex_renderer import LatexRenderer

        logit1 = ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.0, se=0.3, t_stat=3.33,
                    pvalue=0.001, ci_lower=0.4, ci_upper=1.6
                ),
            ],
            n_obs=150,
            n_params=1,
            df_resid=149,
            pseudo_r_squared=0.10,
            aic=180.0,
            bic=185.0,
            llr=15.0,
            dep_var="y",
        )
        logit2 = ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=1.5, se=0.4, t_stat=3.75,
                    pvalue=0.0002, ci_lower=0.7, ci_upper=2.3
                ),
                CoefficientRow(
                    name="x2", coef=-0.3, se=0.15, t_stat=-2.0,
                    pvalue=0.045, ci_lower=-0.6, ci_upper=0.0
                ),
            ],
            n_obs=150,
            n_params=2,
            df_resid=148,
            pseudo_r_squared=0.18,
            aic=160.0,
            bic=168.0,
            llr=22.0,
            dep_var="y",
        )

        latex = LatexRenderer.render_comparison(
            [logit1, logit2],
            model_labels=["Base", "Extended"],
        )
        assert "Model type" in latex
        assert "Logit" in latex
        assert "Pseudo-$R^2$" in latex
        assert "LR $\\chi^2$" in latex
        # Should NOT show F-statistic or R² for all-logit comparison
        # (F-statistic is only shown for OLS or mixed; all-logit uses LR chi2)
        assert "F-statistic" not in latex
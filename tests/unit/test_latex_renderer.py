"""单元测试：LaTeX 表格渲染器。

测试 LatexRenderer 的单模型表和多模型对比表生成功能。
"""

from __future__ import annotations

import pytest

from src.export.latex_renderer import LatexRenderer
from src.results.table import CoefficientRow, ModelResult

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def single_model() -> ModelResult:
    """创建一个简单的单模型结果。"""
    return ModelResult(
        model_type="OLS",
        coefficients=[
            CoefficientRow(
                name="Intercept", coef=2.0, se=0.5, t_stat=4.0,
                pvalue=0.001, ci_lower=1.0, ci_upper=3.0,
            ),
            CoefficientRow(
                name="x1", coef=0.5, se=0.1, t_stat=5.0,
                pvalue=0.0001, ci_lower=0.3, ci_upper=0.7,
            ),
            CoefficientRow(
                name="x2", coef=-0.3, se=0.15, t_stat=-2.0,
                pvalue=0.048, ci_lower=-0.6, ci_upper=-0.01,
            ),
        ],
        n_obs=100,
        n_params=3,
        df_resid=97,
        r_squared=0.85,
        adj_r_squared=0.84,
        rmse=0.5,
        dep_var="y",
        aic=100.0,
        bic=105.0,
        f_statistic=(25.0, 0.0001),
    )


@pytest.fixture
def multi_models() -> list[ModelResult]:
    """创建两个模型用于对比测试。"""
    m1 = ModelResult(
        model_type="OLS",
        coefficients=[
            CoefficientRow(
                name="Intercept", coef=2.0, se=0.5, t_stat=4.0,
                pvalue=0.001, ci_lower=1.0, ci_upper=3.0,
            ),
            CoefficientRow(
                name="x1", coef=0.5, se=0.1, t_stat=5.0,
                pvalue=0.0001, ci_lower=0.3, ci_upper=0.7,
            ),
        ],
        n_obs=100, n_params=2, df_resid=98,
        r_squared=0.85, adj_r_squared=0.84, rmse=0.5,
        dep_var="y", aic=100.0, bic=105.0,
        f_statistic=(25.0, 0.0001),
    )
    m2 = ModelResult(
        model_type="OLS",
        coefficients=[
            CoefficientRow(
                name="Intercept", coef=1.5, se=0.6, t_stat=2.5,
                pvalue=0.014, ci_lower=0.3, ci_upper=2.7,
            ),
            CoefficientRow(
                name="x1", coef=0.6, se=0.12, t_stat=5.0,
                pvalue=0.0001, ci_lower=0.36, ci_upper=0.84,
            ),
            CoefficientRow(
                name="x2", coef=-0.2, se=0.1, t_stat=-2.0,
                pvalue=0.048, ci_lower=-0.4, ci_upper=-0.01,
            ),
        ],
        n_obs=100, n_params=3, df_resid=97,
        r_squared=0.88, adj_r_squared=0.87, rmse=0.4,
        dep_var="y", aic=95.0, bic=102.0,
        f_statistic=(30.0, 0.00001),
    )
    return [m1, m2]


# =========================================================================
# Test: render_single
# =========================================================================


class TestRenderSingle:
    """测试单模型 LaTeX 表格生成。"""

    def test_render_single_returns_string(self, single_model: ModelResult) -> None:
        """应返回非空字符串。"""
        result = LatexRenderer.render_single(single_model)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_single_contains_tabular(self, single_model: ModelResult) -> None:
        """应包含 tabular 环境。"""
        result = LatexRenderer.render_single(single_model)
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result

    def test_render_single_contains_booktabs(self, single_model: ModelResult) -> None:
        """应包含 booktabs 命令。"""
        result = LatexRenderer.render_single(single_model)
        assert r"\toprule" in result
        assert r"\midrule" in result
        assert r"\bottomrule" in result

    def test_render_single_contains_coefficient_names(self, single_model: ModelResult) -> None:
        """应包含所有变量名。"""
        result = LatexRenderer.render_single(single_model)
        assert "Intercept" in result
        assert "x1" in result
        assert "x2" in result

    def test_render_single_contains_significance(self, single_model: ModelResult) -> None:
        """应包含显著性星号。"""
        result = LatexRenderer.render_single(single_model)
        assert "***" in result
        assert "**" in result

    def test_render_single_with_title_caption(self, single_model: ModelResult) -> None:
        """带 title 和 caption 时应包含 table 环境。"""
        result = LatexRenderer.render_single(
            single_model, title="Test Title", caption="Test Caption",
        )
        assert r"\begin{table}" in result
        assert r"\caption{ Test Caption }" in result

    def test_render_single_footer_stats(self, single_model: ModelResult) -> None:
        """应包含拟合统计量（N, R², 调整R², F统计量）。"""
        result = LatexRenderer.render_single(single_model)
        assert "N" in result
        assert "R$^2$" in result or "R\\textsuperscript{2}" in result
        assert "100" in result
        assert "0.85" in result

    def test_render_single_f_statistic(self, single_model: ModelResult) -> None:
        """应包含 F 统计量及其 p 值。"""
        result = LatexRenderer.render_single(single_model)
        assert "25.0000" in result
        assert "0.0001" in result

    def test_render_single_significance_footnote(self, single_model: ModelResult) -> None:
        """应包含显著性脚注。"""
        result = LatexRenderer.render_single(single_model)
        assert "p<0.01" in result
        assert "p<0.05" in result
        assert "p<0.1" in result

    def test_render_single_empty(self) -> None:
        """空模型（无系数）应仍能生成基本结构。"""
        empty_result = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0, n_params=0, df_resid=0,
        )
        result = LatexRenderer.render_single(empty_result)
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result


# =========================================================================
# Test: render_comparison
# =========================================================================


class TestRenderComparison:
    """测试多模型对比 LaTeX 表格生成。"""

    def test_render_comparison_returns_string(self, multi_models: list[ModelResult]) -> None:
        """应返回非空字符串。"""
        result = LatexRenderer.render_comparison(multi_models)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_comparison_contains_tabular(self, multi_models: list[ModelResult]) -> None:
        """应包含 tabular 环境。"""
        result = LatexRenderer.render_comparison(multi_models)
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result

    def test_render_comparison_model_labels(self, multi_models: list[ModelResult]) -> None:
        """应包含默认模型标签（Model 1, Model 2）。"""
        result = LatexRenderer.render_comparison(multi_models)
        assert "Model 1" in result
        assert "Model 2" in result

    def test_render_comparison_custom_labels(self, multi_models: list[ModelResult]) -> None:
        """应使用自定义标签。"""
        labels = ["Baseline", "Full"]
        result = LatexRenderer.render_comparison(multi_models, model_labels=labels)
        assert "Baseline" in result
        assert "Full" in result

    def test_render_comparison_all_variables(self, multi_models: list[ModelResult]) -> None:
        """应包含所有变量名（跨模型并集）。"""
        result = LatexRenderer.render_comparison(multi_models)
        assert "Intercept" in result
        assert "x1" in result
        assert "x2" in result

    def test_render_comparison_fit_stats(self, multi_models: list[ModelResult]) -> None:
        """应包含拟合统计量。"""
        result = LatexRenderer.render_comparison(multi_models)
        assert "R$^2$" in result
        assert "AIC" in result
        assert "BIC" in result
        assert "0.85" in result
        assert "0.88" in result

    def test_render_comparison_empty_list(self) -> None:
        """空列表应返回空字符串。"""
        result = LatexRenderer.render_comparison([])
        assert result == ""

    def test_render_comparison_single_model(self, single_model: ModelResult) -> None:
        """单模型列表也能正确生成。"""
        result = LatexRenderer.render_comparison([single_model])
        assert r"\begin{tabular}" in result
        assert "Model 1" in result

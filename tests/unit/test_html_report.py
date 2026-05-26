"""单元测试：HTML 报告生成器。

测试 HtmlReportGenerator 的自包含 HTML 报告生成功能。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.export.html_report import HtmlReportGenerator
from src.results.table import CoefficientRow, ModelResult

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def model_result() -> ModelResult:
    """创建一个模型结果用于测试。"""
    return ModelResult(
        model_type="OLS",
        coefficients=[
            CoefficientRow(
                name="Intercept", coef=2.0, se=0.5, t_stat=4.0,
                pvalue=0.001, ci_lower=1.0, ci_upper=3.0,
                significance="***",
            ),
            CoefficientRow(
                name="x1", coef=0.5, se=0.1, t_stat=5.0,
                pvalue=0.0001, ci_lower=0.3, ci_upper=0.7,
                significance="***",
            ),
            CoefficientRow(
                name="x2", coef=-0.3, se=0.15, t_stat=-2.0,
                pvalue=0.048, ci_lower=-0.6, ci_upper=-0.01,
                significance="**",
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
        specification="y ~ x1 + x2",
    )


@pytest.fixture
def data_summary() -> pd.DataFrame:
    """创建一个简单描述性统计表。"""
    return pd.DataFrame({
        "观测数": [100, 100],
        "均值": [3.5, 15.2],
        "标准差": [1.2, 5.1],
        "最小值": [1.0, 5.0],
        "最大值": [6.0, 25.0],
    }, index=pd.Index(["y", "x1"], name="变量"))


# =========================================================================
# Test: generate_full_report
# =========================================================================


class TestGenerateFullReport:
    """测试完整 HTML 报告生成。"""

    def test_report_returns_string(self, model_result: ModelResult) -> None:
        """应返回非空字符串。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_report_doctype(self, model_result: ModelResult) -> None:
        """应以 DOCTYPE 开头。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert html.startswith("<!DOCTYPE html>") or html.startswith("<!DOCTYPE html")

    def test_report_contains_title(self, model_result: ModelResult) -> None:
        """应包含报告标题。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "回归分析报告" in html

    def test_report_contains_timestamp(self, model_result: ModelResult) -> None:
        """应包含时间戳。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "生成时间" in html

    def test_report_contains_model_spec(self, model_result: ModelResult) -> None:
        """应包含模型规格。"""
        html = HtmlReportGenerator.generate_full_report(
            None, model_result, model_spec="y ~ x1 + x2"
        )
        assert "模型设定" in html
        assert "y ~ x1 + x2" in html

    def test_report_contains_descriptive_stats(
        self, model_result: ModelResult, data_summary: pd.DataFrame
    ) -> None:
        """应包含描述性统计表。"""
        html = HtmlReportGenerator.generate_full_report(data_summary, model_result)
        assert "描述性统计" in html
        assert "观测数" in html
        assert "均值" in html
        assert "标准差" in html

    def test_report_contains_coefficient_table(self, model_result: ModelResult) -> None:
        """应包含回归系数表。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "回归结果" in html
        assert "Intercept" in html
        assert "x1" in html
        assert "x2" in html

    def test_report_contains_fit_stats(self, model_result: ModelResult) -> None:
        """应包含模型拟合统计量。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "模型拟合统计量" in html
        assert "R²" in html
        assert "0.8500" in html
        assert "AIC" in html
        assert "RMSE" in html

    def test_report_contains_notes(self, model_result: ModelResult) -> None:
        """应包含注意事项部分。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "注意事项" in html

    def test_report_contains_dep_var(self, model_result: ModelResult) -> None:
        """应包含因变量名。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "y" in html

    def test_report_self_contained_css(self, model_result: ModelResult) -> None:
        """应包含内联 CSS 样式（无外部引用）。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "font-family" in html
        assert "<style>" in html
        assert "</style>" in html

    def test_report_no_external_links(self, model_result: ModelResult) -> None:
        """不应包含外部资源链接。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert 'src="http' not in html

    def test_report_footer(self, model_result: ModelResult) -> None:
        """应包含页脚。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result)
        assert "Regression Analysis Tool" in html

    def test_report_minimal_model(self) -> None:
        """最小模型应仍能生成报告。"""
        minimal = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0, n_params=0, df_resid=0,
        )
        html = HtmlReportGenerator.generate_full_report(None, minimal)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_report_chart_placeholder(self, model_result: ModelResult) -> None:
        """无图表时应显示占位文本。"""
        html = HtmlReportGenerator.generate_full_report(None, model_result, charts_dict={})
        assert "诊断图" in html

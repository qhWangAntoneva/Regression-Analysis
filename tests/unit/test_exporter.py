# encoding: utf-8
"""单元测试：数据导出模块。

测试 DataExporter 类的各导出方法。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_io.exporter import DataExporter, _get_model_summary


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """创建一个简单的 DataFrame 用于导出测试。"""
    return pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "y": [3.5, 5.2, 7.1, 8.8, 10.5],
        "category": ["A", "B", "A", "C", "B"],
    })


@pytest.fixture
def temp_dir() -> str:
    """创建一个临时目录用于导出。"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # 清理
    for f in Path(tmpdir).glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
    try:
        Path(tmpdir).rmdir()
    except Exception:
        pass


# =========================================================================
# Test: export_csv
# =========================================================================


class TestExportCSV:
    """测试 CSV 导出功能。"""

    def test_export_csv_basic(self, sample_df: pd.DataFrame, temp_dir: str) -> None:
        """基本 CSV 导出。"""
        filepath = os.path.join(temp_dir, "test.csv")
        result = DataExporter.export_csv(sample_df, filepath)
        assert os.path.exists(result)
        assert result == os.path.abspath(filepath)

        # 验证内容
        loaded = pd.read_csv(result, encoding="utf-8-sig")
        assert loaded.shape == sample_df.shape
        assert list(loaded.columns) == list(sample_df.columns)

    def test_export_csv_utf8_bom(self, sample_df: pd.DataFrame, temp_dir: str) -> None:
        """CSV 导出应包含 UTF-8 BOM 以支持 Excel 打开。"""
        filepath = os.path.join(temp_dir, "utf8_test.csv")
        DataExporter.export_csv(sample_df, filepath)

        with open(filepath, "rb") as f:
            raw = f.read(10)
        # 检查 BOM (0xEF 0xBB 0xBF)
        assert raw[:3] == b"\xef\xbb\xbf"

    def test_export_csv_empty(self, temp_dir: str) -> None:
        """空 DataFrame 应抛出 ValueError。"""
        empty_df = pd.DataFrame()
        filepath = os.path.join(temp_dir, "empty.csv")
        with pytest.raises(ValueError, match="为空"):
            DataExporter.export_csv(empty_df, filepath)

    def test_export_csv_none(self, temp_dir: str) -> None:
        """None 应抛出 ValueError。"""
        filepath = os.path.join(temp_dir, "none.csv")
        with pytest.raises(ValueError, match="为空"):
            DataExporter.export_csv(None, filepath)  # type: ignore[arg-type]


# =========================================================================
# Test: export_excel
# =========================================================================


class TestExportExcel:
    """测试 Excel 导出功能。"""

    def test_export_excel_basic(self, sample_df: pd.DataFrame, temp_dir: str) -> None:
        """基本 Excel 导出。"""
        filepath = os.path.join(temp_dir, "test.xlsx")
        result = DataExporter.export_excel(sample_df, filepath)
        assert os.path.exists(result)

        # 验证内容
        loaded = pd.read_excel(result, engine="openpyxl")
        assert loaded.shape == sample_df.shape

    def test_export_excel_custom_sheet(self, sample_df: pd.DataFrame, temp_dir: str) -> None:
        """自定义工作表名称。"""
        filepath = os.path.join(temp_dir, "custom.xlsx")
        result = DataExporter.export_excel(sample_df, filepath, sheet_name="回归结果")

        loaded = pd.read_excel(result, engine="openpyxl")
        assert loaded is not None

    def test_export_excel_empty(self, temp_dir: str) -> None:
        """空 DataFrame 应抛出 ValueError。"""
        empty_df = pd.DataFrame()
        filepath = os.path.join(temp_dir, "empty.xlsx")
        with pytest.raises(ValueError, match="为空"):
            DataExporter.export_excel(empty_df, filepath)


# =========================================================================
# Test: export_chart
# =========================================================================


class TestExportChart:
    """测试图表导出功能（plotly）。"""

    def test_export_chart_png(self, temp_dir: str) -> None:
        """导出 plotly 图表为 PNG。"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly 未安装")

        fig = go.Figure()
        fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])

        filepath = os.path.join(temp_dir, "chart.png")
        result = DataExporter.export_chart(fig, filepath)
        assert os.path.exists(result)
        assert result.endswith(".png")
        assert os.path.getsize(result) > 0

    def test_export_chart_svg(self, temp_dir: str) -> None:
        """导出 plotly 图表为 SVG。"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly 未安装")

        fig = go.Figure()
        fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])

        filepath = os.path.join(temp_dir, "chart.svg")
        result = DataExporter.export_chart(fig, filepath, format="svg")
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_chart_invalid_format(self, temp_dir: str) -> None:
        """不支持的格式应抛出 ValueError。"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly 未安装")

        fig = go.Figure()
        filepath = os.path.join(temp_dir, "chart.bmp")
        with pytest.raises(ValueError, match="不支持的格式"):
            DataExporter.export_chart(fig, filepath, format="bmp")

    def test_export_chart_none(self, temp_dir: str) -> None:
        """None 图表应抛出 ValueError。"""
        filepath = os.path.join(temp_dir, "none.png")
        with pytest.raises(ValueError, match="为空"):
            DataExporter.export_chart(None, filepath)  # type: ignore[arg-type]

    def test_export_chart_custom_dimensions(self, temp_dir: str) -> None:
        """自定义尺寸和 DPI。"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly 未安装")

        fig = go.Figure()
        fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])

        filepath = os.path.join(temp_dir, "custom.png")
        result = DataExporter.export_chart(fig, filepath, width=800, height=600, dpi=150)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0


# =========================================================================
# Test: export_comparison_table
# =========================================================================


class TestExportComparisonTable:
    """测试对比表导出功能。"""

    def test_export_comparison_csv(self, temp_dir: str) -> None:
        """导出对比表为 CSV。"""
        comparison = pd.DataFrame({
            "模型": ["模型 1", "模型 2"],
            "R²": [0.85, 0.90],
            "AIC": [100.0, 95.0],
        })
        filepath = os.path.join(temp_dir, "comparison.csv")
        result = DataExporter.export_comparison_table(comparison, filepath, format="csv")
        assert os.path.exists(result)

        loaded = pd.read_csv(result, encoding="utf-8-sig")
        assert len(loaded) == 2

    def test_export_comparison_excel(self, temp_dir: str) -> None:
        """导出对比表为 Excel。"""
        comparison = pd.DataFrame({
            "模型": ["模型 1", "模型 2"],
            "R²": [0.85, 0.90],
        })
        filepath = os.path.join(temp_dir, "comparison.xlsx")
        result = DataExporter.export_comparison_table(comparison, filepath, format="excel")
        assert os.path.exists(result)

    def test_export_comparison_invalid_format(self, temp_dir: str) -> None:
        """不支持的格式应抛出 ValueError。"""
        comparison = pd.DataFrame({"a": [1]})
        filepath = os.path.join(temp_dir, "bad.txt")
        with pytest.raises(ValueError, match="不支持的格式"):
            DataExporter.export_comparison_table(comparison, filepath, format="txt")

    def test_export_comparison_empty(self, temp_dir: str) -> None:
        """空表应抛出 ValueError。"""
        empty = pd.DataFrame()
        filepath = os.path.join(temp_dir, "empty.csv")
        with pytest.raises(ValueError, match="为空"):
            DataExporter.export_comparison_table(empty, filepath)


# =========================================================================
# Test: export_results_package
# =========================================================================


class TestExportResultsPackage:
    """测试完整结果包导出功能。"""

    def test_export_results_package_basic(self, sample_df: pd.DataFrame, temp_dir: str) -> None:
        """基本结果包导出。"""
        from src.results.table import CoefficientRow, ModelResult

        # 构造一个简单的 ModelResult
        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(name="Intercept", coef=2.0, se=0.5, t_stat=4.0, pvalue=0.001, ci_lower=1.0, ci_upper=3.0),
                CoefficientRow(name="x1", coef=0.5, se=0.1, t_stat=5.0, pvalue=0.0001, ci_lower=0.3, ci_upper=0.7),
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.85,
            adj_r_squared=0.84,
            rmse=0.5,
            dep_var="y",
            specification="y ~ x1",
        )

        coef_df = result.to_dataframe().reset_index()

        # 空图表字典
        chart_figs: dict = {}

        prefix = os.path.join(temp_dir, "results/test_run")
        exported = DataExporter.export_results_package(result, coef_df, chart_figs, prefix)

        # 应导出系数表 CSV 和 Excel 以及摘要 TXT
        assert "coefficients_csv" in exported
        assert "coefficients_xlsx" in exported
        assert "summary_txt" in exported

        # 验证文件存在
        assert os.path.exists(exported["coefficients_csv"])
        assert os.path.exists(exported["summary_txt"])

    def test_export_results_package_with_charts(self, temp_dir: str) -> None:
        """导出带图表的结果包。"""
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(name="Intercept", coef=2.0, se=0.5, t_stat=4.0, pvalue=0.001, ci_lower=1.0, ci_upper=3.0),
                CoefficientRow(name="x1", coef=0.5, se=0.1, t_stat=5.0, pvalue=0.0001, ci_lower=0.3, ci_upper=0.7),
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.85,
            rmse=5.0,
            dep_var="y",
            specification="y ~ x1",
        )

        coef_df = result.to_dataframe().reset_index()

        try:
            import plotly.graph_objects as go

            chart_figs = {
                "residual_fitted": go.Figure(),
                "qq": go.Figure(),
            }
            chart_figs["residual_fitted"].add_scatter(x=[1, 2, 3], y=[4, 5, 6])
            chart_figs["qq"].add_scatter(x=[1, 2, 3], y=[4, 5, 6])
        except ImportError:
            chart_figs = {}

        prefix = os.path.join(temp_dir, "results/with_charts")
        exported = DataExporter.export_results_package(result, coef_df, chart_figs, prefix)

        assert "coefficients_csv" in exported
        assert "summary_txt" in exported

    def test_export_results_package_empty_result(self, temp_dir: str) -> None:
        """导出应处理无效结果而不崩溃。"""
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0,
            n_params=0,
            df_resid=0,
        )

        empty_df = pd.DataFrame()
        prefix = os.path.join(temp_dir, "results/empty")
        exported = DataExporter.export_results_package(result, empty_df, {}, prefix)

        # 即使数据无效也应返回字典
        assert isinstance(exported, dict)


# =========================================================================
# Test: helper functions
# =========================================================================


class TestModelSummary:
    """测试模型摘要文本生成。"""

    def test_summary_basic(self) -> None:
        """基本摘要文本应包含关键统计量。"""
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(name="Intercept", coef=2.0, se=0.5, t_stat=4.0, pvalue=0.001, ci_lower=1.0, ci_upper=3.0),
                CoefficientRow(name="x1", coef=0.5, se=0.1, t_stat=5.0, pvalue=0.0001, ci_lower=0.3, ci_upper=0.7),
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            r_squared=0.85,
            adj_r_squared=0.84,
            rmse=0.5,
            dep_var="y",
            specification="y ~ x1",
            aic=100.0,
            bic=105.0,
            method="OLS",
            log_likelihood=-50.0,
            f_statistic=(25.0, 0.0001),
        )

        summary = _get_model_summary(result)
        assert "R²" in summary or "R-squared" in summary
        assert "0.85" in summary
        assert "y" in summary
        assert "OLS" in summary

    def test_summary_minimal(self) -> None:
        """最小 ModelResult 也应生成摘要。"""
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0,
            n_params=0,
            df_resid=0,
        )

        summary = _get_model_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

# encoding: utf-8
"""
数据/结果导出模块

提供 DataExporter 类，支持导出 DataFrame 为 CSV/Excel、
导出图表为图片、导出完整结果包等。
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[assignment]


class DataExporter:
    """数据/结果导出工具类。

    提供静态方法，支持：
    - DataFrame 导出为 CSV、Excel
    - 图表导出为 PNG、SVG、PDF
    - 完整结果包导出（系数表 + 图表 + 模型摘要）
    """

    @staticmethod
    def export_csv(data: pd.DataFrame, filepath: str) -> str:
        """导出 DataFrame 为 CSV（UTF-8 编码）。

        Args:
            data: 要导出的 DataFrame。
            filepath: 输出文件路径。

        Returns:
            写入的文件路径。

        Raises:
            ValueError: data 为空。
            IOError: 文件写入失败。
        """
        if data is None or data.empty:
            raise ValueError("DataFrame 为空，无法导出")

        try:
            data.to_csv(filepath, index=False, encoding="utf-8-sig")
        except OSError as e:
            raise IOError(f"CSV 文件写入失败: {e}") from e

        return os.path.abspath(filepath)

    @staticmethod
    def export_excel(
        data: pd.DataFrame,
        filepath: str,
        sheet_name: str = "Sheet1",
    ) -> str:
        """导出 DataFrame 为 Excel (.xlsx)。

        Args:
            data: 要导出的 DataFrame。
            filepath: 输出文件路径。
            sheet_name: Excel 工作表名称。

        Returns:
            写入的文件路径。

        Raises:
            ValueError: data 为空。
            ImportError: openpyxl 或 xlsxwriter 未安装。
            IOError: 文件写入失败。
        """
        if data is None or data.empty:
            raise ValueError("DataFrame 为空，无法导出")

        try:
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                data.to_excel(writer, sheet_name=sheet_name, index=False)
        except ImportError as e:
            raise ImportError(
                "导出 Excel 需要安装 openpyxl: pip install openpyxl"
            ) from e
        except OSError as e:
            raise IOError(f"Excel 文件写入失败: {e}") from e

        return os.path.abspath(filepath)

    @staticmethod
    def export_chart(
        fig: Any,
        filepath: str,
        format: str = "png",
        dpi: int = 300,
        width: int = 1200,
        height: int = 800,
    ) -> str:
        """导出图表为图片文件。

        支持 plotly Figure 和 matplotlib Figure。
        - plotly: 使用 plotly.io.write_image + kaleido
        - matplotlib: 使用 fig.savefig
        - 支持的格式: png, svg, pdf

        Args:
            fig: plotly Figure 或 matplotlib Figure 对象。
            filepath: 输出文件路径。
            format: 图片格式（'png', 'svg', 'pdf'），默认 'png'。
            dpi: 图片 DPI，默认 300（满足出版要求）。
            width: 图片宽度（像素），默认 1200。
            height: 图片高度（像素），默认 800。

        Returns:
            写入的文件路径。

        Raises:
            ValueError: fig 为空或格式不支持。
            IOError: 文件写入失败。
        """
        if fig is None:
            raise ValueError("图表对象为空，无法导出")

        valid_formats = {"png", "svg", "pdf"}
        if format not in valid_formats:
            raise ValueError(
                f"不支持的格式 '{format}'。支持: {', '.join(valid_formats)}"
            )

        try:
            # 检测是否为 plotly Figure
            if _is_plotly_figure(fig):
                # plotly 导出需要 kaleido 或 orca
                fig.write_image(
                    filepath,
                    format=format,
                    width=width,
                    height=height,
                    scale=dpi / 100,  # plotly 的默认基准是 100 DPI
                )
            elif _is_matplotlib_figure(fig):
                # matplotlib 导出
                fig.savefig(
                    filepath,
                    dpi=dpi,
                    format=format,
                    bbox_inches="tight",
                )
            else:
                raise ValueError(
                    "不支持的图表类型。支持 plotly Figure 和 matplotlib Figure。"
                )
        except (ImportError, ValueError) as e:
            raise ImportError(
                f"图表导出失败。可能需要安装 kaleido: pip install kaleido\n原始错误: {e}"
            ) from e
        except OSError as e:
            raise IOError(f"图表文件写入失败: {e}") from e

        return os.path.abspath(filepath)

    @staticmethod
    def export_results_package(
        result: Any,
        coefficient_df: pd.DataFrame,
        chart_figs: dict[str, Any],
        filepath_prefix: str,
    ) -> dict[str, str]:
        """导出完整结果包。

        包含：
        - 系数表 (CSV + Excel)
        - 各诊断图 (PNG)
        - 模型摘要文本 (TXT)

        Args:
            result: ModelResult 对象（用于生成摘要文本）。
            coefficient_df: 系数表 DataFrame。
            chart_figs: 图表字典 {图表名: Figure}。
            filepath_prefix: 输出文件路径前缀（不含扩展名）。

        Returns:
            字典 {文件类型: 文件路径}。
        """
        exported: dict[str, str] = {}
        prefix = str(filepath_prefix)

        # 确保输出目录存在
        output_dir = Path(prefix).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 系数表 CSV
        try:
            csv_path = f"{prefix}_coefficients.csv"
            DataExporter.export_csv(coefficient_df, csv_path)
            exported["coefficients_csv"] = csv_path
        except Exception as e:
            exported["coefficients_csv"] = f"导出失败: {e}"

        # 2. 系数表 Excel
        try:
            xlsx_path = f"{prefix}_coefficients.xlsx"
            DataExporter.export_excel(coefficient_df, xlsx_path)
            exported["coefficients_xlsx"] = xlsx_path
        except Exception as e:
            exported["coefficients_xlsx"] = f"导出失败: {e}"

        # 3. 各诊断图
        for fig_name, fig in chart_figs.items():
            try:
                chart_path = f"{prefix}_{fig_name}.png"
                DataExporter.export_chart(fig, chart_path)
                exported[f"chart_{fig_name}"] = chart_path
            except Exception as e:
                exported[f"chart_{fig_name}"] = f"导出失败: {e}"

        # 4. 模型摘要文本
        try:
            txt_path = f"{prefix}_summary.txt"
            summary = _get_model_summary(result)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(summary)
            exported["summary_txt"] = txt_path
        except Exception as e:
            exported["summary_txt"] = f"导出失败: {e}"

        return exported

    @staticmethod
    def export_comparison_table(
        comparison_df: pd.DataFrame,
        filepath: str,
        format: str = "csv",
    ) -> str:
        """导出多模型对比表。

        Args:
            comparison_df: 对比表 DataFrame。
            filepath: 输出文件路径。
            format: 导出格式（'csv' 或 'excel'）。

        Returns:
            写入的文件路径。

        Raises:
            ValueError: comparison_df 为空或格式不支持。
        """
        if comparison_df is None or comparison_df.empty:
            raise ValueError("对比表为空，无法导出")

        if format == "csv":
            return DataExporter.export_csv(comparison_df, filepath)
        elif format == "excel":
            return DataExporter.export_excel(comparison_df, filepath)
        else:
            raise ValueError(f"不支持的格式 '{format}'。支持: 'csv', 'excel'")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_plotly_figure(fig: Any) -> bool:
    """检测对象是否为 plotly Figure。"""
    if go is None:
        return False
    return isinstance(fig, go.Figure)


def _is_matplotlib_figure(fig: Any) -> bool:
    """检测对象是否为 matplotlib Figure。"""
    try:
        import matplotlib.figure

        return isinstance(fig, matplotlib.figure.Figure)
    except ImportError:
        return False


def _get_model_summary(result: Any) -> str:
    """从 ModelResult 对象生成模型摘要文本。"""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  回归模型结果摘要")
    lines.append("=" * 60)
    lines.append("")

    # 基本信息
    dep_var = getattr(result, "dep_var", "")
    spec = getattr(result, "specification", "")
    model_type = getattr(result, "model_type", "OLS")
    method = getattr(result, "method", "OLS")

    lines.append(f"  因变量 (Dependent Variable):  {dep_var}")
    lines.append(f"  模型公式:                     {spec}")
    lines.append(f"  估计方法:                     {method}")
    lines.append("")

    # 模型统计量
    n_obs = getattr(result, "n_obs", None)
    n_params = getattr(result, "n_params", None)
    df_resid = getattr(result, "df_resid", None)
    lines.append(f"  观测数 (No. Observations):    {n_obs}" if n_obs is not None else "")
    lines.append(f"  参数数 (No. Parameters):      {n_params}" if n_params is not None else "")
    lines.append(f"  残差自由度 (Residual DF):     {df_resid}" if df_resid is not None else "")
    lines.append("")

    rsq = getattr(result, "r_squared", None)
    rsq_adj = getattr(result, "adj_r_squared", None)
    rmse = getattr(result, "rmse", None)
    if rsq is not None:
        lines.append(f"  R² (R-squared):               {rsq:.6f}")
    if rsq_adj is not None:
        lines.append(f"  调整 R² (Adj. R-squared):     {rsq_adj:.6f}")
    if rmse is not None:
        lines.append(f"  RMSE:                         {rmse:.6f}")
    lines.append("")

    f_stat = getattr(result, "f_statistic", None)
    if f_stat is not None and len(f_stat) == 2:
        lines.append(f"  F 统计量 (F-statistic):       {f_stat[0]:.4f}")
        lines.append(f"  F 检验 p 值:                  {f_stat[1]:.6e}")
    lines.append("")

    log_likelihood = getattr(result, "log_likelihood", None)
    aic = getattr(result, "aic", None)
    bic = getattr(result, "bic", None)
    if log_likelihood is not None:
        lines.append(f"  对数似然 (Log-Likelihood):    {log_likelihood:.4f}")
    if aic is not None:
        lines.append(f"  AIC:                          {aic:.4f}")
    if bic is not None:
        lines.append(f"  BIC:                          {bic:.4f}")
    lines.append("")

    # 系数表
    coefficients = getattr(result, "coefficients", None)
    if coefficients:
        lines.append("-" * 60)
        lines.append(f"  {'变量':<20} {'系数':>12} {'标准误':>10} {'t值':>8} {'p值':>8}")
        lines.append("-" * 60)
        for c in coefficients:
            name = getattr(c, "name", "?")
            coef = getattr(c, "coef", 0.0)
            se = getattr(c, "se", 0.0)
            t_stat = getattr(c, "t_stat", 0.0)
            pvalue = getattr(c, "pvalue", 1.0)
            sig = getattr(c, "significance", "")
            lines.append(f"  {name:<20} {coef:>12.6f} {se:>10.6f} {t_stat:>8.4f} {pvalue:>8.4f} {sig}")

        lines.append("-" * 60)
        lines.append("  显著性标记: *** p<0.01, ** p<0.05, * p<0.1")

    lines.append("")
    lines.append("=" * 60)
    lines.append("  导出时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 60)

    return "\n".join(line for line in lines if line)  # 移除空行

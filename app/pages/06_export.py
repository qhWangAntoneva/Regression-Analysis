"""
导出与报告页面

提供数据导出、结果导出、图表导出和一键综合报告功能。
使用 st.download_button 实现浏览器下载。
新增: LaTeX 表格导出、HTML 报告导出、分析复现包导出。
"""  # noqa: N999

from __future__ import annotations

import io
import tempfile
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports
EXPORTER_AVAILABLE = False
try:
    from src.data_io.exporter import DataExporter

    EXPORTER_AVAILABLE = True
except ImportError:
    pass

PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]

LATEX_AVAILABLE = False
try:
    from src.export.latex_renderer import LatexRenderer

    LATEX_AVAILABLE = True
except ImportError:
    pass

HTML_REPORT_AVAILABLE = False
try:
    from src.export.html_report import HtmlReportGenerator

    HTML_REPORT_AVAILABLE = True
except ImportError:
    pass


def render() -> None:
    """渲染导出与报告页面。"""
    if st is None:
        return

    st.title(":material/download: 导出与报告")

    # 检查是否有数据
    df = st.session_state.get("data")
    model_result = st.session_state.get("model_result")

    if df is None and model_result is None:
        st.info("暂无数据可导出。请先上传数据并运行回归模型。")
        st.page_link("app/pages/01_data_upload.py", label="前往数据上传", icon="📂")
        return

    # =========================================================================
    # 数据导出
    # =========================================================================
    st.subheader("数据导出")

    if df is not None:
        data_col1, data_col2, data_col3 = st.columns(3)

        with data_col1:
            csv_bytes = _df_to_csv_bytes(df)
            st.download_button(
                label=":material/table: 下载 CSV",
                data=csv_bytes,
                file_name="regression_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with data_col2:
            xlsx_bytes = _df_to_excel_bytes(df)
            if xlsx_bytes is not None:
                st.download_button(
                    label=":material/table: 下载 Excel",
                    data=xlsx_bytes,
                    file_name="regression_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.warning("Excel 导出需要 openpyxl")

        with data_col3:
            st.caption(f"数据集: {st.session_state.get('filename', '未命名')}")
            st.caption(f"行数: {len(df)}, 列数: {len(df.columns)}")

        st.success("数据导出成功！" if df is not None else "")
    else:
        st.info("当前无数据集。")

    st.divider()

    # =========================================================================
    # 结果导出
    # =========================================================================
    st.subheader("结果导出")

    if model_result is not None:
        # 获取系数表
        coefficient_df = _get_coefficient_dataframe(model_result)
        summary_text = _get_summary_text(model_result)

        if coefficient_df is not None:
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                coef_csv = _df_to_csv_bytes(coefficient_df)
                st.download_button(
                    label=":material/table: 下载系数表 (CSV)",
                    data=coef_csv,
                    file_name="coefficients.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with res_col2:
                coef_xlsx = _df_to_excel_bytes(coefficient_df, sheet_name="系数表")
                if coef_xlsx is not None:
                    st.download_button(
                        label=":material/table: 下载系数表 (Excel)",
                        data=coef_xlsx,
                        file_name="coefficients.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                else:
                    st.info("Excel 导出需要 openpyxl")

        if summary_text:
            summary_bytes = summary_text.encode("utf-8")
            st.download_button(
                label=":material/article: 下载模型摘要 (TXT)",
                data=summary_bytes,
                file_name="model_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.success("结果导出成功！")
    else:
        st.info("当前无模型结果。请在「模型设定」页面运行回归。")

    st.divider()

    # =========================================================================
    # 图表导出
    # =========================================================================
    st.subheader("图表导出")

    # 从 session_state 获取暂存的图表
    export_charts = st.session_state.get("export_charts", {})

    if export_charts:
        st.caption("以下图表来自回归结果页面的暂存")

        chart_display_names = {
            "residual_vs_fitted": "残差 vs 拟合值图",
            "qq_plot": "正态 Q-Q 图",
            "scale_location": "尺度-位置图",
            "cooks_distance": "Cook's Distance 图",
            "系数图": "系数图",
        }

        for fig_name, fig in export_charts.items():
            display_name = chart_display_names.get(fig_name, fig_name)
            with st.container(border=True):
                st.markdown(f"**{display_name}**")

                col1, col2 = st.columns(2)
                with col1:
                    _render_download_chart_button(fig, f"{fig_name}.png", "PNG")
                with col2:
                    _render_download_chart_button(fig, f"{fig_name}.svg", "SVG")
    else:
        st.info("暂无暂存图表。请先在「回归结果」页面查看诊断图并点击「保存为 PNG」按钮。")

        # 提供直接在当前页面生成图表的选项
        if model_result is not None and PLOTLY_AVAILABLE:
            with st.expander("在当前页面生成诊断图", expanded=False):
                if st.button("生成诊断图并导出", use_container_width=True):
                    _generate_charts_from_result(model_result, df)

    st.divider()

    # =========================================================================
    # 综合报告
    # =========================================================================
    st.subheader("综合报告")

    if model_result is not None and coefficient_df is not None:
        if st.button(
            ":material/package_2: 一键导出所有结果",
            type="primary",
            use_container_width=True,
        ):
            _export_all_results(model_result, coefficient_df, export_charts, summary_text)
    else:
        st.info("需要模型结果才能生成综合报告。")

    st.divider()

    # =========================================================================
    # LaTeX 表格导出（Phase 3.2）
    # =========================================================================
    st.subheader("LaTeX 表格导出")

    if model_result is not None:
        if LATEX_AVAILABLE:
            latex_tab = st.tabs(["单个模型表格", "LaTeX 预览"])

            with latex_tab[0]:
                latex_single = LatexRenderer.render_single(
                    model_result,
                    title="回归结果",
                    caption="回归分析结果表",
                    label="regression",
                )
                st.download_button(
                    label=":material/code: 下载 LaTeX (.tex)",
                    data=latex_single.encode("utf-8"),
                    file_name="regression_table.tex",
                    mime="text/plain",
                    use_container_width=True,
                )

                # 检查是否有多个模型结果
                model_results_list = st.session_state.get("model_results_list", None)
                if model_results_list and len(model_results_list) > 1:
                    latex_compare = LatexRenderer.render_comparison(
                        model_results_list,
                        captions=["多模型对比"],
                    )
                    st.download_button(
                        label=":material/compare_arrows: 下载对比表格 (.tex)",
                        data=latex_compare.encode("utf-8"),
                        file_name="model_comparison.tex",
                        mime="text/plain",
                        use_container_width=True,
                    )

            with latex_tab[1]:
                st.code(latex_single, language="latex", line_numbers=True)
        else:
            st.info("LaTeX 导出需要 jinja2 库。")

        st.success("LaTeX 表格已生成！")
    else:
        st.info("需要模型结果才能生成 LaTeX 表格。")

    st.divider()

    # =========================================================================
    # HTML 报告导出（Phase 3.2）
    # =========================================================================
    st.subheader("HTML 报告导出")

    if model_result is not None:
        if HTML_REPORT_AVAILABLE:
            if st.button(
                ":material/language: 生成 HTML 报告",
                type="primary",
                use_container_width=True,
            ):
                # Collect data summary
                data_summary = None
                if df is not None:
                    try:
                        from src.results.statistics import descriptive_stats

                        coef_names = _get_coefficient_variable_names(model_result)
                        all_vars = _collect_all_variables(model_result, df)
                        if all_vars:
                            data_summary = descriptive_stats(df, all_vars)
                    except Exception:
                        pass

                # Model spec text
                model_spec_text = getattr(model_result, "specification", "")

                # Charts
                charts = st.session_state.get("export_charts", {})

                html_content = HtmlReportGenerator.generate_full_report(
                    data_summary=data_summary,
                    model_result=model_result,
                    charts_dict=charts,
                    model_spec=model_spec_text,
                )

                st.download_button(
                    label=":material/download: 下载 HTML 报告",
                    data=html_content.encode("utf-8"),
                    file_name="regression_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

                st.success("HTML 报告已生成！")
        else:
            st.info("HTML 报告导出需要 jinja2 库。")
    else:
        st.info("需要模型结果才能生成 HTML 报告。")

    st.divider()

    # =========================================================================
    # 分析复现包导出（Phase 3.2）
    # =========================================================================
    st.subheader("分析复现包导出")

    if model_result is not None and df is not None:
        if EXPORTER_AVAILABLE:
            if st.button(
                ":material/science: 生成分析复现包 (ZIP)",
                type="primary",
                use_container_width=True,
            ):
                # Build a model_spec-like object from session state
                model_spec = st.session_state.get("model_spec", None)
                if model_spec is None:
                    # Fallback: construct a simple spec from model result
                    from dataclasses import dataclass

                    @dataclass
                    class FallbackSpec:
                        dep_var: str
                        indep_vars: list
                        control_vars: list
                        has_intercept: bool

                    dep_var = getattr(model_result, "dep_var", "")
                    coef_names = _get_coefficient_variable_names(model_result)
                    model_spec = FallbackSpec(
                        dep_var=dep_var,
                        indep_vars=coef_names,
                        control_vars=[],
                        has_intercept=True,
                    )

                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        zip_path = DataExporter.export_reproducibility_package(
                            data=df,
                            model_spec=model_spec,
                            model_result=model_result,
                            export_dir=tmpdir,
                        )
                        with open(zip_path, "rb") as f:
                            zip_bytes = f.read()

                        st.download_button(
                            label=":material/download: 下载复现包 (ZIP)",
                            data=zip_bytes,
                            file_name="reproducibility_package.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                        st.success("分析复现包已生成！")
                    except Exception as e:
                        st.error(f"复现包生成失败: {e}")
        else:
            st.info("分析复现包导出需要 DataExporter。")
    else:
        st.info("需要模型结果和数据才能生成分析复现包。")

    st.divider()

    # =========================================================================
    # 页脚
    # =========================================================================
    st.caption(
        "Regression Analysis Tool v0.2.0 | "
        "导出支持: CSV, Excel, PNG, SVG, LaTeX, HTML, ZIP | "
        "数据不会被上传到任何服务器"
    )


# =========================================================================
# Helper functions
# =========================================================================


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """将 DataFrame 转换为 CSV 字节流。"""
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return buf.getvalue()


def _df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes | None:
    """将 DataFrame 转换为 Excel 字节流。"""
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        buf.seek(0)
        return buf.getvalue()
    except ImportError:
        return None


def _get_coefficient_dataframe(result: Any) -> pd.DataFrame | None:
    """从模型结果提取系数表 DataFrame。"""
    if hasattr(result, "to_dataframe"):
        try:
            df = result.to_dataframe()
            return df.reset_index() if df is not None else None
        except Exception:
            pass

    coefficients = getattr(result, "coefficients", None)
    if not coefficients:
        return None

    rows = []
    for c in coefficients:
        name = getattr(c, "name", "?")
        coef = getattr(c, "coef", None) or getattr(c, "coefficient", None) or 0.0
        se = getattr(c, "se", None) or getattr(c, "std_err", None) or 0.0
        t_stat = getattr(c, "t_stat", None) or getattr(c, "t_statistic", None) or 0.0
        p_val = getattr(c, "pvalue", None) or getattr(c, "p_value", None) or 1.0
        ci_low = getattr(c, "ci_lower", None) or 0.0
        ci_high = getattr(c, "ci_upper", None) or 0.0

        sig = ""
        if p_val <= 0.01:
            sig = "***"
        elif p_val <= 0.05:
            sig = "**"
        elif p_val <= 0.1:
            sig = "*"

        rows.append(
            {
                "变量": name,
                "系数": coef,
                "标准误": se,
                "t值": t_stat,
                "p值": p_val,
                "95%CI下限": ci_low,
                "95%CI上限": ci_high,
                "显著性": sig,
            }
        )

    return pd.DataFrame(rows)


def _get_summary_text(result: Any) -> str:
    """获取模型摘要文本。"""
    if hasattr(result, "summary"):
        try:
            return result.summary()
        except Exception:
            pass

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  回归模型结果摘要")
    lines.append("=" * 60)
    lines.append("")

    dep_var = getattr(result, "dep_var", "N/A")
    model_type = getattr(result, "model_type", "OLS")
    lines.append(f"  因变量:  {dep_var}")
    lines.append(f"  模型类型: {model_type}")
    lines.append("")

    for attr, label in [
        ("n_obs", "观测数"),
        ("n_params", "参数数"),
        ("df_resid", "残差自由度"),
    ]:
        val = getattr(result, attr, None)
        if val is not None:
            lines.append(f"  {label}: {val}")

    lines.append("")
    for attr, label, fmt in [
        ("r_squared", "R²", ".6f"),
        ("adj_r_squared", "调整 R²", ".6f"),
        ("rmse", "RMSE", ".6f"),
        ("aic", "AIC", ".4f"),
        ("bic", "BIC", ".4f"),
    ]:
        val = getattr(result, attr, None)
        if val is not None:
            lines.append(f"  {label}: {val:{fmt}}")

    return "\n".join(lines)


def _render_download_chart_button(fig: Any, filename: str, img_format: str) -> None:
    """渲染图表下载按钮。"""
    if st is None or not EXPORTER_AVAILABLE:
        return

    try:
        img_bytes = _fig_to_bytes(fig, img_format.lower())
        if img_bytes is not None:
            mime_map = {"png": "image/png", "svg": "image/svg+xml"}
            st.download_button(
                label=f"下载 {img_format}",
                data=img_bytes,
                file_name=filename,
                mime=mime_map.get(img_format.lower(), "image/png"),
                use_container_width=True,
            )
    except Exception:
        st.info(f"{img_format} 导出不可用（需要 kaleido）")


def _fig_to_bytes(fig: Any, fmt: str = "png") -> bytes | None:
    """将图表转换为字节流。"""
    try:
        import plotly.io as pio

        img_bytes = pio.write_image(
            fig,
            format=fmt,
            width=1200,
            height=800,
            scale=3 if fmt == "png" else 1,
        )
        return img_bytes
    except Exception:
        return None


def _generate_charts_from_result(result: Any, df: Any) -> None:
    """从模型结果生成诊断图并保存到 session_state。"""
    from src.visualization.residual import (
        cooks_distance_plot,
        qq_plot,
        residual_vs_fitted_plot,
        scale_location_plot,
    )

    if "export_charts" not in st.session_state:
        st.session_state.export_charts = {}

    try:
        st.session_state.export_charts["residual_vs_fitted"] = residual_vs_fitted_plot(result, df)
    except Exception:
        pass

    residuals = getattr(result, "residuals", None)
    if residuals is not None:
        try:
            st.session_state.export_charts["qq_plot"] = qq_plot(residuals)
        except Exception:
            pass

    try:
        st.session_state.export_charts["scale_location"] = scale_location_plot(result, df)
    except Exception:
        pass

    try:
        st.session_state.export_charts["cooks_distance"] = cooks_distance_plot(result, df)
    except Exception:
        pass

    st.success("图表已生成！请在上方下载区域查看。")
    st.rerun()


def _export_all_results(
    result: Any,
    coefficient_df: pd.DataFrame,
    export_charts: dict[str, Any],
    summary_text: str,
) -> None:
    """一键导出所有结果为 ZIP 包。"""
    import zipfile

    if st is None:
        return

    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 系数表 CSV
        csv_str = coefficient_df.to_csv(index=False, encoding="utf-8-sig")
        zf.writestr("coefficients.csv", csv_str)

        # 系数表 Excel
        try:
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                coefficient_df.to_excel(writer, sheet_name="系数表", index=False)
            xlsx_buf.seek(0)
            zf.writestr("coefficients.xlsx", xlsx_buf.getvalue())
        except Exception:
            pass

        # 模型摘要
        zf.writestr("model_summary.txt", summary_text.encode("utf-8"))

        # 图表
        for fig_name, fig in export_charts.items():
            try:
                img_bytes = _fig_to_bytes(fig, "png")
                if img_bytes is not None:
                    zf.writestr(f"charts/{fig_name}.png", img_bytes)
            except Exception:
                pass

    buf.seek(0)
    st.download_button(
        label=":material/download: 下载综合报告 (ZIP)",
        data=buf.getvalue(),
        file_name="regression_report.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.success("综合报告已生成！点击上方按钮下载 ZIP 包。")


# ---------------------------------------------------------------------------
# Phase 3.2 helpers
# ---------------------------------------------------------------------------


def _get_coefficient_variable_names(result: Any) -> list[str]:
    """从模型结果提取系数变量名列表。"""
    coefficients = getattr(result, "coefficients", None)
    if not coefficients:
        return []
    names = []
    for c in coefficients:
        name = getattr(c, "name", "")
        if name and name.lower() not in ("const", "intercept"):
            names.append(name)
    return names


def _collect_all_variables(result: Any, df: Any) -> list[str]:
    """收集模型中使用的所有变量（用于描述性统计）。"""
    if df is None:
        return []

    dep_var = getattr(result, "dep_var", "")
    spec = getattr(result, "specification", "")

    # Try to get from model spec
    model_spec = getattr(result, "model_spec", None)
    if model_spec is not None:
        all_vars = getattr(model_spec, "all_predictors", None)
        if all_vars:
            return [dep_var] + list(all_vars) if dep_var else list(all_vars)

    # Fallback: extract from specification string
    if spec and "~" in spec:
        parts = spec.split("~")
        rhs = parts[-1].strip() if len(parts) > 1 else ""
        rhs_vars = [v.strip() for v in rhs.split("+") if v.strip()]
        # Filter out patsy C() wrappers
        clean_vars = []
        for v in rhs_vars:
            if v.startswith("C("):
                inner = v[2:-1].strip()
                clean_vars.append(inner)
            else:
                clean_vars.append(v)
        result_vars = [dep_var] + clean_vars if dep_var else clean_vars
        # Only include columns that actually exist in df
        if df is not None:
            result_vars = [v for v in result_vars if v in df.columns]
        return result_vars

    # Last resort: coefficient names
    coef_names = _get_coefficient_variable_names(result)
    result_vars = [dep_var] + coef_names if dep_var else coef_names
    if df is not None:
        result_vars = [v for v in result_vars if v in df.columns]
    return result_vars


# 页面入口
render()

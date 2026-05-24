# encoding: utf-8
"""Result display UI components for regression model output.

Provides Streamlit-based rendering of coefficient tables, model statistics,
ANOVA tables, comparison tables, and statistical alerts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.results.table import CoefficientRow, ModelResult

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


def render_coefficient_table(
    result: ModelResult,
    use_stars: bool = True,
) -> None:
    """Render a formatted coefficient table with significance highlighting.

    Args:
        result: A ModelResult object.
        use_stars: If True, show significance stars column.
    """
    if st is None:
        return

    df = result.to_dataframe().reset_index()

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        p_val = row["p值"]
        sig = row.get("显著性", "") if use_stars else ""
        ci_str = f"[{row['95%CI低']:.4f}, {row['95%CI高']:.4f}]"

        rows.append(
            {
                "变量": row["变量"],
                "系数(B)": round(float(row["系数"]), 6),
                "标准误": round(float(row["标准误"]), 6),
                "t值": round(float(row["t值"]), 4),
                "p值": float(p_val) if isinstance(p_val, (int, float)) else p_val,
                "95% CI": ci_str,
                "显著性": sig,
            }
        )

    display_df = pd.DataFrame(rows)

    # Apply conditional highlighting
    def _highlight_rows(row_series: pd.Series) -> List[str]:
        styles: List[str] = [""] * len(row_series)
        p_val = row_series.get("p值", 1.0)
        if isinstance(p_val, (int, float)):
            if p_val < 0.01:
                styles = ["background-color: #c8e6c9"] * len(row_series)  # dark green
            elif p_val < 0.05:
                styles = ["background-color: #e8f5e9"] * len(row_series)  # light green
        return styles

    styled = display_df.style.apply(_highlight_rows, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "变量": st.column_config.TextColumn("变量"),
            "系数(B)": st.column_config.NumberColumn("系数(B)", format="%.6f"),
            "标准误": st.column_config.NumberColumn("标准误", format="%.6f"),
            "t值": st.column_config.NumberColumn("t值", format="%.4f"),
            "p值": st.column_config.NumberColumn("p值", format="%.6f"),
            "95% CI": st.column_config.TextColumn("95% CI"),
            "显著性": st.column_config.TextColumn("显著性"),
        },
        hide_index=True,
    )

    st.caption("* p<0.1, ** p<0.05, *** p<0.01")
    st.caption("绿色背景行表示 p<0.05；深绿色背景行表示 p<0.01")


def render_model_statistics(result: ModelResult) -> None:
    """Render model statistics as a grid of metric cards.

    Displays R-squared, Adj-R-squared, RMSE, AIC, BIC, Log-Likelihood,
    F-statistic, F-p-value, and N in a 3x3 grid.

    Args:
        result: A ModelResult object.
    """
    if st is None:
        return

    summary = result.to_summary_dict()

    # Row 1: R-squared metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        r2 = summary.get("r_squared")
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        r2_delta = None
        if r2 is not None and r2 < 0.1:
            r2_delta = "模型解释力较低"
        st.metric(
            label="R²",
            value=r2_str,
            delta=r2_delta,
            delta_color="inverse" if r2_delta else "normal",
        )

    with col2:
        adj_r2 = summary.get("adj_r_squared")
        st.metric(
            label="Adj-R²",
            value=f"{adj_r2:.4f}" if adj_r2 is not None else "N/A",
        )

    with col3:
        rmse = summary.get("rmse")
        st.metric(label="RMSE", value=f"{rmse:.4f}" if rmse else "N/A")

    # Row 2: Information criteria
    col1, col2, col3 = st.columns(3)
    with col1:
        aic = summary.get("aic")
        st.metric(label="AIC", value=f"{aic:.2f}" if aic else "N/A")

    with col2:
        bic = summary.get("bic")
        st.metric(label="BIC", value=f"{bic:.2f}" if bic else "N/A")

    with col3:
        ll = summary.get("log_likelihood")
        st.metric(
            label="Log-Likelihood",
            value=f"{ll:.4f}" if ll is not None else "N/A",
        )

    # Row 3: F-test and sample size
    col1, col2, col3 = st.columns(3)
    with col1:
        f_stat = summary.get("f_statistic")
        st.metric(
            label="F 统计量",
            value=f"{f_stat:.4f}" if f_stat is not None else "N/A",
        )

    with col2:
        f_p = summary.get("f_pvalue")
        f_p_str = f"{f_p:.6f}" if f_p is not None else "N/A"
        st.metric(label="F-p值", value=f_p_str)

    with col3:
        n = summary.get("n_obs")
        st.metric(label="N (样本量)", value=str(n) if n else "N/A")

    # Warning for low R-squared
    if r2 is not None and r2 < 0.1:
        st.warning(
            ":material/warning: R² = {:.4f}，模型解释力较低。".format(r2)
        )


def render_anova_table(result: ModelResult) -> None:
    """Render the ANOVA (analysis of variance) table.

    Args:
        result: A ModelResult object.
    """
    if st is None:
        return

    try:
        anova_df = result.anova_table()
        st.dataframe(
            anova_df,
            use_container_width=True,
            column_config={
                "来源": st.column_config.TextColumn("来源"),
                "SS": st.column_config.NumberColumn("SS", format="%.6f"),
                "df": st.column_config.NumberColumn("df", format="%d"),
                "MS": st.column_config.NumberColumn("MS", format="%.6f"),
                "F": st.column_config.NumberColumn("F", format="%.6f"),
                "p-value": st.column_config.NumberColumn("p-value", format="%.6f"),
            },
            hide_index=True,
        )
    except Exception as e:
        st.error(f"ANOVA 表生成失败: {e}")


def render_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Render a horizontal multi-model comparison table.

    Args:
        comparison_df: DataFrame from ``compare_models()``.
    """
    if st is None:
        return

    if comparison_df.empty:
        st.info("没有对比数据。")
        return

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )


def render_statistical_alerts(
    result: ModelResult,
    vif_df: Optional[pd.DataFrame] = None,
    residual_tests: Optional[Dict[str, Any]] = None,
) -> None:
    """Render statistical assumption violation alerts.

    Checks for:
      - Multicollinearity (VIF > 10: severe warning; > 5: mild warning).
      - Influential observations (Cook's distance > 4/n).
      - Durbin-Watson autocorrelation (if residual tests provided).

    Args:
        result: A ModelResult object.
        vif_df: DataFrame from ``diagnostics.vif()``, or None.
        residual_tests: Dict from ``diagnostics.residual_tests()``, or None.
    """
    if st is None:
        return

    alerts: List[str] = []
    warnings_list: List[str] = []

    # --- VIF alerts ---
    if vif_df is not None and not vif_df.empty:
        for _, row in vif_df.iterrows():
            var_name = str(row.get("variable", ""))
            vif_val = row.get("vif", 0)
            diagnosis = str(row.get("diagnosis", ""))
            # Skip constant/intercept
            if var_name.lower() in ("const", "intercept"):
                continue
            if vif_val > 10 or diagnosis == "High":
                alerts.append(
                    f":material/dangerous: **{var_name}** 存在严重多重共线性"
                    f"(VIF = {vif_val:.2f})"
                )
            elif vif_val > 5 or diagnosis == "Moderate":
                warnings_list.append(
                    f":material/warning: **{var_name}** 存在中等多重共线性"
                    f"(VIF = {vif_val:.2f})"
                )

    # --- Influential observations (Cook's distance) ---
    try:
        # Try to compute Cook's distance from residuals (approximate)
        # The engine stores residuals only if available
        if hasattr(result, "residuals") and result.residuals is not None:
            residuals = np.asarray(result.residuals).flatten()
            n = len(residuals)
            if n > 0:
                # Approximate Cook's distance threshold: 4/n
                threshold = 4.0 / n
                # Count observations exceeding threshold using leverage approx
                # Simple heuristic: squared standardized residual > threshold
                std_resid = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)
                influence_count = int(np.sum(std_resid ** 2 > threshold))
                if influence_count > 0:
                    warnings_list.append(
                        f":material/info: 发现 {influence_count} 个高影响力观测点"
                        f"(Cook's distance > 4/n)"
                    )
    except Exception:
        pass

    # --- Durbin-Watson autocorrelation ---
    if residual_tests is not None:
        dw_auto = residual_tests.get("dw_autocorrelation", "")
        if dw_auto and dw_auto not in ("None", "Insufficient data", ""):
            dw_stat = residual_tests.get("dw_stat", float("nan"))
            if "strong" in dw_auto.lower():
                alerts.append(
                    f":material/dangerous: 残差存在强烈的自相关性"
                    f"(Durbin-Watson = {dw_stat:.4f}, {dw_auto})"
                )
            else:
                warnings_list.append(
                    f":material/warning: 残差存在轻微的自相关性"
                    f"(Durbin-Watson = {dw_stat:.4f}, {dw_auto})"
                )

    # Render alerts
    if alerts:
        for alert in alerts:
            st.error(alert, icon=":material/error:")

    if warnings_list:
        for warning in warnings_list:
            st.warning(warning, icon=":material/warning:")

    if not alerts and not warnings_list:
        st.success(
            ":material/check_circle: 未发现明显的统计假设违规。",
            icon=":material/check_circle:",
        )

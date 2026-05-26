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

try:
    from app.config import get_color_scheme
except ImportError:
    get_color_scheme = None  # type: ignore[assignment]


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

    is_mle = getattr(result, "is_mle_model", False)
    is_binary = getattr(result, "is_binary_choice", False)
    df = result.to_dataframe().reset_index()

    # Use human-readable variable labels if available
    variable_labels: Dict[str, str] = getattr(result, "variable_labels", {}) or {}

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        p_val = row["p值"]
        sig = row.get("显著性", "") if use_stars else ""
        ci_str = f"[{row['95%CI低']:.4f}, {row['95%CI高']:.4f}]"

        stat_col = "z值" if is_mle else "t值"
        stat_val = row.get(stat_col, row.get("t值", row.get("z值", 0)))

        raw_name = str(row["变量"])
        display_name = variable_labels.get(raw_name, raw_name)
        row_data = {
            "变量": display_name,
            "系数(B)": round(float(row["系数"]), 6),
            "标准误": round(float(row["标准误"]), 6),
            stat_col: round(float(stat_val), 4),
            "p值": float(p_val) if isinstance(p_val, (int, float)) else p_val,
            "95% CI": ci_str,
            "显著性": sig,
        }

        # Add OR column for binary choice models
        if is_binary:
            import math
            or_val = math.exp(float(row["系数"]))
            row_data["OR (几率比)"] = round(or_val, 4)

        rows.append(row_data)

    display_df = pd.DataFrame(rows)

    # Load color scheme for accessibility support
    colors = get_color_scheme() if get_color_scheme is not None else {
        "sig_high_bg": "#c8e6c9",
        "sig_med_bg": "#e8f5e9",
    }

    # Apply conditional highlighting
    def _highlight_rows(row_series: pd.Series) -> list[str]:
        styles: list[str] = [""] * len(row_series)
        p_val = row_series.get("p值", 1.0)
        if isinstance(p_val, (int, float)):
            if p_val < 0.01:
                styles = [f"background-color: {colors['sig_high_bg']}"] * len(row_series)
            elif p_val < 0.05:
                styles = [f"background-color: {colors['sig_med_bg']}"] * len(row_series)
        return styles

    styled = display_df.style.apply(_highlight_rows, axis=1)

    stat_col_label = "z值" if is_mle else "t值"
    col_config = {
        "变量": st.column_config.TextColumn("变量"),
        "系数(B)": st.column_config.NumberColumn("系数(B)", format="%.6f"),
        "标准误": st.column_config.NumberColumn("标准误", format="%.6f"),
        stat_col_label: st.column_config.NumberColumn(stat_col_label, format="%.4f"),
        "p值": st.column_config.NumberColumn("p值", format="%.6f"),
        "95% CI": st.column_config.TextColumn("95% CI"),
        "显著性": st.column_config.TextColumn("显著性"),
    }
    if is_binary:
        col_config["OR (几率比)"] = st.column_config.NumberColumn("OR (几率比)", format="%.4f")

    st.dataframe(
        styled,
        use_container_width=True,
        column_config=col_config,
        hide_index=True,
    )

    st.caption("* p<0.1, ** p<0.05, *** p<0.01")
    if is_binary:
        st.caption("OR (几率比) = exp(系数)，表示自变量每增加一个单位，因变量发生概率的倍率变化。")
    # Annotation uses correct color description based on active palette
    if get_color_scheme is not None and st.session_state.get("colorblind_mode", False):
        st.caption("蓝色背景行表示 p<0.05；深蓝色背景行表示 p<0.01（色盲友好）")
    else:
        st.caption("绿色背景行表示 p<0.05；深绿色背景行表示 p<0.01")

    # SE type annotation
    se_type = getattr(result, "se_type", "nonrobust")
    if is_mle:
        st.caption("MLE 模型使用最大似然估计，标准误为渐近标准误。")
    elif se_type and se_type != "nonrobust":
        st.caption(f"使用稳健标准误: {se_type}")
    else:
        st.caption("使用普通标准误")

    # Transformed variable annotation
    transforms = getattr(result, "transforms_applied", {})
    if transforms:
        parts = [f"{t}({v})" for v, t in transforms.items()]
        st.caption("已转换变量: " + ", ".join(parts))

    # Interaction term annotation
    interactions = getattr(result, "interaction_terms_applied", [])
    if interactions:
        parts = [f"{v1}:{v2}" for v1, v2 in interactions]
        st.caption("交互项: " + ", ".join(parts))


def render_model_statistics(result: ModelResult) -> None:
    """Render model statistics as a grid of metric cards.

    Displays R-squared/Pseudo R-squared, Adj-R-squared/LR chi2, RMSE/Pseudo R²,
    AIC, BIC, Log-Likelihood, F-statistic/LR chi2, p-value, and N in a 3x3 grid.

    For logit models, OLS-specific metrics (R², Adj-R², RMSE, F-test) are
    replaced with logit counterparts (Pseudo R², LR chi2, LR p-value).

    Args:
        result: A ModelResult object.
    """
    if st is None:
        return

    summary = result.to_summary_dict()
    is_mle = getattr(result, "is_mle_model", False)

    # Row 1: Model fit metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_mle:
            pr2 = summary.get("pseudo_r_squared")
            pr2_str = f"{pr2:.4f}" if pr2 is not None else "N/A"
            pr2_delta = None
            if pr2 is not None and pr2 < 0.1:
                pr2_delta = "模型解释力较低"
            st.metric(
                label="伪 R² (McFadden)",
                value=pr2_str,
                delta=pr2_delta,
                delta_color="inverse" if pr2_delta else "normal",
            )
        else:
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
        if is_mle:
            llr_val = summary.get("llr")
            st.metric(
                label="似然比检验 (LR χ²)",
                value=f"{llr_val:.4f}" if llr_val is not None else "N/A",
            )
        else:
            adj_r2 = summary.get("adj_r_squared")
            st.metric(
                label="Adj-R²",
                value=f"{adj_r2:.4f}" if adj_r2 is not None else "N/A",
            )

    with col3:
        if is_mle:
            llr_p = summary.get("llr_pvalue")
            st.metric(
                label="LR p值",
                value=f"{llr_p:.6f}" if llr_p is not None else "N/A",
            )
        else:
            rmse = summary.get("rmse")
            st.metric(label="RMSE", value=f"{rmse:.4f}" if rmse else "N/A")

    # Row 2: Information criteria (same for both)
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

    # Row 3: Test statistics (OLS) or sample info (logit)
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_mle:
            st.metric(label="样本量 (N)", value=f"{summary.get('n_obs', 'N/A')}")
        else:
            f_stat = summary.get("f_statistic")
            st.metric(
                label="F 统计量",
                value=f"{f_stat:.4f}" if f_stat is not None else "N/A",
            )

    with col2:
        if is_mle:
            pass  # Row 1 already shows LR χ² and p-value
        else:
            f_p = summary.get("f_pvalue")
            f_p_str = f"{f_p:.6f}" if f_p is not None else "N/A"
            st.metric(label="F-p值", value=f_p_str)

    with col3:
        n = summary.get("n_obs")
        st.metric(label="N (样本量)", value=str(n) if n else "N/A")

    # Warning for low fit
    if is_mle:
        pr2_warn = summary.get("pseudo_r_squared")
        if pr2_warn is not None and pr2_warn < 0.1:
            st.warning(
                ":material/warning: 伪 R² = {:.4f}，模型解释力较低。".format(pr2_warn)
            )
    else:
        r2_warn = summary.get("r_squared")
        if r2_warn is not None and r2_warn < 0.1:
            st.warning(
                ":material/warning: R² = {:.4f}，模型解释力较低。".format(r2_warn)
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

    # SE type info
    se_type = getattr(result, "se_type", "nonrobust")
    if se_type and se_type != "nonrobust":
        st.caption(f"标准误类型: {se_type}（异方差稳健）")
    else:
        st.caption("标准误类型: 普通标准误（假设同方差）")

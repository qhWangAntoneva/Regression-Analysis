"""
回归结果展示页面

Streamlit 页面：系数表、模型统计量、多模型对比、诊断图、统计警示。
"""  # noqa: N999

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]
    px = None  # type: ignore[assignment]

VIS_AVAILABLE = False
try:
    from src.visualization.residual import (
        cooks_distance_plot,
        diagnostic_dashboard,
        qq_plot,
        residual_vs_fitted_plot,
        scale_location_plot,
    )
    from src.visualization.scatter import scatter_with_regression

    VIS_AVAILABLE = True
except ImportError:
    pass

COEFF_VIS_AVAILABLE = False
try:
    from src.visualization.coefficient import coefficient_plot_single

    COEFF_VIS_AVAILABLE = True
except ImportError:
    pass

# Phase 2 components
RESULT_CARD_AVAILABLE = False
try:
    from app.components.result_card import (
        render_anova_table,
        render_coefficient_table,
        render_comparison_table,
        render_model_statistics,
        render_statistical_alerts,
    )

    RESULT_CARD_AVAILABLE = True
except ImportError:
    pass

SUMMARY_GEN_AVAILABLE = False
try:
    from src.results.summary_generator import (
        generate_assumption_check_text,
        generate_coefficient_interpretation,  # noqa: F401
        generate_summary_text,
    )

    SUMMARY_GEN_AVAILABLE = True
except ImportError:
    pass

DIAGNOSTICS_AVAILABLE = False
try:
    from src.modeling.diagnostics import residual_tests, vif

    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    pass

COMPARE_AVAILABLE = False
try:
    from src.results.table import compare_models

    COMPARE_AVAILABLE = True
except ImportError:
    pass


def render() -> None:
    """渲染回归结果页面。"""
    if st is None:
        return

    st.title(":material/bar_chart: 回归结果")

    # Gallery mode detection
    if st.session_state.get("gallery_mode") and st.session_state.get("gallery_item_title"):
        st.info(
            f":material/science: **示例数据，仅供功能演示** — "
            f"{st.session_state.gallery_item_title}"
        )
        from src.utils.gallery import get_gallery_item
        gallery_item = get_gallery_item(st.session_state.get("gallery_item_id", ""))
        if gallery_item and gallery_item.story:
            with st.expander(":material/auto_stories: 分析场景说明", expanded=False):
                st.markdown(gallery_item.story)
        if gallery_item and gallery_item.key_features:
            with st.expander(":material/info: 本场景特点", expanded=False):
                for feat in gallery_item.key_features:
                    st.markdown(f"- {feat}")

    # 检查模型结果是否存在
    model_result = st.session_state.get("model_result")
    if model_result is None and not st.session_state.get("model_results_list"):
        st.warning("请先在「模型设定」页面运行回归模型。")
        st.page_link("app/pages/03_model_spec.py", label="前往模型设定", icon="⚙️")
        return

    df = st.session_state.get("data")
    variables = st.session_state.get("variables")  # noqa: F841
    model_spec = st.session_state.get("model_spec")
    model_config = st.session_state.get("model_config", {})  # noqa: F841

    # --- Phase 2: 多模型选择 ---
    results_list: list[Any] = st.session_state.get("model_results_list", [])
    selected_result: Any = model_result

    if len(results_list) > 1:
        selected_result = _render_model_selector(results_list)
    elif len(results_list) == 1:
        selected_result = results_list[0]
        model_spec = st.session_state.get("model_spec")
    else:
        # Fallback to single model
        selected_result = model_result
        results_list = [model_result] if model_result else []

    if selected_result is None:
        st.info("无可用模型结果。")
        return

    # ------------------------------------------------------------------
    # Phase 3.1: 转换/交互项/SE 类型信息
    # ------------------------------------------------------------------
    transforms_applied = getattr(selected_result, "transforms_applied", {})
    interaction_terms_applied = getattr(
        selected_result, "interaction_terms_applied", []
    )
    se_type = getattr(selected_result, "se_type", "nonrobust")

    if transforms_applied:
        parts = [f"{t}({v})" for v, t in transforms_applied.items()]
        st.info(f"已应用变量转换: {', '.join(parts)}")

    if interaction_terms_applied:
        parts = [f"{v1}:{v2}" for v1, v2 in interaction_terms_applied]
        st.info(f"已添加交互项: {', '.join(parts)}")

    se_label = "普通标准误" if se_type == "nonrobust" else f"{se_type} (异方差稳健)"
    st.caption(f"标准误类型: {se_label}")

    # ------------------------------------------------------------------
    # Phase 2: 模型摘要文本
    # ------------------------------------------------------------------
    if SUMMARY_GEN_AVAILABLE:
        try:
            summary_text = generate_summary_text(selected_result)
            with st.expander("模型摘要", expanded=False):
                st.markdown(summary_text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 2: 模型统计量卡片 (replaces old inline metrics)
    # ------------------------------------------------------------------
    st.subheader("模型统计量")

    if RESULT_CARD_AVAILABLE:
        render_model_statistics(selected_result)
    else:
        _fallback_model_statistics(selected_result)

    # ------------------------------------------------------------------
    # Phase 2: 系数表 (replaces old _render_coefficient_table)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("系数表")

    if RESULT_CARD_AVAILABLE and hasattr(selected_result, "coefficients"):
        render_coefficient_table(selected_result)
    else:
        _fallback_coefficient_table(selected_result)

    # ------------------------------------------------------------------
    # Phase 2: ANOVA 表 (OLS only)
    # ------------------------------------------------------------------
    st.divider()

    is_mle = getattr(selected_result, "is_mle_model", False)

    if is_mle:
        st.subheader("方差分析(ANOVA)表")
        st.info("MLE 模型使用最大似然估计，不使用 ANOVA 平方和分解。"
                "请参考上方似然比检验 (LR χ²) 来评估模型整体显著性。")
    else:
        st.subheader("方差分析(ANOVA)表")
        if RESULT_CARD_AVAILABLE:
            render_anova_table(selected_result)
        else:
            st.info("ANOVA 表组件不可用。")

    # ------------------------------------------------------------------
    # Phase 2: 统计警示区
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("统计诊断")

    vif_df = None
    rt_results = None

    # Compute VIF
    if DIAGNOSTICS_AVAILABLE and model_spec is not None and df is not None:
        try:
            vif_df = vif(df, model_spec, use_patsy=True)
        except Exception:
            pass

    # Compute residual tests
    if DIAGNOSTICS_AVAILABLE:
        try:
            residuals = getattr(selected_result, "residuals", None)
            if residuals is not None:
                rt_results = residual_tests(np.asarray(residuals).flatten())
            else:
                # Try to extract from the underlying statsmodels object
                raw_model = getattr(selected_result, "_raw_model", None)
                if raw_model is not None:
                    rt_results = residual_tests(
                        np.asarray(raw_model.resid).flatten()
                    )
        except Exception:
            pass

    if RESULT_CARD_AVAILABLE:
        render_statistical_alerts(
            selected_result, vif_df=vif_df, residual_tests=rt_results
        )
    else:
        st.info("统计诊断组件不可用。")

    # ------------------------------------------------------------------
    # Phase 2: 假设检验摘要
    # ------------------------------------------------------------------
    if SUMMARY_GEN_AVAILABLE:
        try:
            assumption_text = generate_assumption_check_text(
                selected_result, vif_df=vif_df, residual_tests=rt_results
            )
            with st.expander("假设检验详细报告", expanded=False):
                st.text(assumption_text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 2: 多模型对比表
    # ------------------------------------------------------------------
    if len(results_list) > 1 and COMPARE_AVAILABLE:
        st.divider()
        st.subheader("多模型对比")
        try:
            comparison_df = compare_models(results_list)
            if RESULT_CARD_AVAILABLE:
                render_comparison_table(comparison_df)
            else:
                st.dataframe(comparison_df, use_container_width=True)
        except Exception as e:
            st.error(f"对比表生成失败: {e}")

    # ------------------------------------------------------------------
    # Phase 2: 系数图 (dot-whisker plot)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("系数可视化")

    if PLOTLY_AVAILABLE and hasattr(selected_result, "coefficients"):
        # 优先使用新的系数图模块
        if COEFF_VIS_AVAILABLE:
            try:
                coef_fig = coefficient_plot_single(selected_result)
                st.plotly_chart(coef_fig, use_container_width=True)
            except Exception:
                _render_dot_whisker_plot(selected_result)
        else:
            _render_dot_whisker_plot(selected_result)

    # ------------------------------------------------------------------
    # 模型公式
    # ------------------------------------------------------------------
    if model_spec:
        formula_str = (
            f"{model_spec.dep_var} ~ "
            f"{' + '.join(model_spec.indep_vars)}"
        )
        with st.expander("模型公式", expanded=False):
            st.code(formula_str, language="text")

    # ------------------------------------------------------------------
    # 诊断图 (existing Phase 1 functionality)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("诊断图")

    if not PLOTLY_AVAILABLE or not VIS_AVAILABLE:
        st.warning("请安装 plotly 和 statsmodels 以显示诊断图。")
        st.code("pip install plotly statsmodels", language="bash")

    if PLOTLY_AVAILABLE and VIS_AVAILABLE:
        # 诊断总览面板
        with st.expander("诊断总览面板 (2x2)", expanded=False):
            _render_diagnostic_dashboard(selected_result, df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "散点图 + 回归线",
                "残差 vs 拟合值",
                "Q-Q 图",
                "尺度-位置图",
                "Cook's Distance",
            ]
        )

        with tab1:
            _render_scatter_plot(selected_result, df, model_spec)

        with tab2:
            _render_residual_plot(selected_result, df)
            if st.button(":material/download: 保存为 PNG", key="save_resid_fitted_04"):
                _save_fig_to_session(selected_result, df, "residual_vs_fitted")
                st.success("图表已暂存，可前往「导出与报告」页面下载。")

        with tab3:
            _render_qq_plot(selected_result)
            if st.button(":material/download: 保存为 PNG", key="save_qq_04"):
                _save_fig_to_session(selected_result, df, "qq_plot")
                st.success("图表已暂存，可前往「导出与报告」页面下载。")

        with tab4:
            _render_scale_location_plot(selected_result, df)
            if st.button(":material/download: 保存为 PNG", key="save_scale_location_04"):
                _save_fig_to_session(selected_result, df, "scale_location")
                st.success("图表已暂存，可前往「导出与报告」页面下载。")

        with tab5:
            _render_cooks_distance_plot(selected_result, df)
            if st.button(":material/download: 保存为 PNG", key="save_cooks_04"):
                _save_fig_to_session(selected_result, df, "cooks_distance")
                st.success("图表已暂存，可前往「导出与报告」页面下载。")


# =========================================================================
# Phase 2 helpers
# =========================================================================


def _render_model_selector(results_list: list[Any]) -> Any:
    """Model selection dropdown for multi-model results.

    Args:
        results_list: List of ModelResult objects.

    Returns:
        The selected ModelResult.
    """
    if st is None:
        return results_list[0] if results_list else None

    options = []
    for i, res in enumerate(results_list):
        label = f"模型 {i+1}"
        if hasattr(res, "specification") and res.specification:
            label += f": {res.specification}"
        options.append(label)

    selected_label = st.selectbox(
        "选择要查看的模型",
        options=options,
        index=0,
        key="result_model_selector",
    )

    idx = options.index(selected_label)
    return results_list[idx]


def _render_dot_whisker_plot(result: Any) -> None:
    """Render a dot-whisker plot of coefficient estimates with confidence intervals.

    Args:
        result: A ModelResult object.
    """
    if st is None or not PLOTLY_AVAILABLE:
        return

    coeffs = getattr(result, "coefficients", None)
    if not coeffs:
        st.info("系数数据不可用，无法绘制系数图。")
        return

    # Filter out Intercept
    filtered = [c for c in coeffs if c.name not in ("Intercept", "const")]

    if not filtered:
        st.info("只有截距项，无法绘制系数图。")
        return

    var_names = [c.name for c in filtered]
    coef_vals = [c.coef for c in filtered]
    ci_lowers = [c.ci_lower for c in filtered]
    ci_uppers = [c.ci_upper for c in filtered]
    p_values = [c.pvalue for c in filtered]

    # Color by significance — use color scheme from config for accessibility
    from app.config import get_color_scheme

    cs = get_color_scheme()
    plot_colors = []
    for p in p_values:
        if p < 0.01:
            plot_colors.append(cs["sig_high"])
        elif p < 0.05:
            plot_colors.append(cs["sig_med"])
        elif p < 0.1:
            plot_colors.append(cs["sig_low"])
        else:
            plot_colors.append(cs["no_sig"])

    fig = go.Figure()

    # Add error bars (CI)
    fig.add_trace(
        go.Scatter(
            x=coef_vals,
            y=var_names,
            mode="markers",
            marker=dict(size=10, color=plot_colors, line=dict(width=1, color="black")),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - v for v, u in zip(coef_vals, ci_uppers)],
                arrayminus=[v - l for v, l in zip(coef_vals, ci_lowers)],  # noqa: E741
                visible=True,
                color="gray",
                thickness=1.5,
                width=3,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "系数: %{x:.4f}<br>"
                "95%% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(ci_lowers, ci_uppers)),
        )
    )

    # Add zero reference line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
    )

    is_binary = getattr(result, "is_binary_choice", False)
    x_label = "系数估计值 (log-odds)" if is_binary else "系数估计值"

    fig.update_layout(
        title="系数点图 (Dot-Whisker Plot)" if not is_binary else "系数点图 (Logit/Probit, Dot-Whisker Plot)",  # noqa: E501
        xaxis_title=x_label,
        yaxis_title="变量",
        template=cs.get("plot_template", "plotly_white"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Fallback rendering (when result_card module is not available)
# =========================================================================


def _fallback_model_statistics(result: Any) -> None:
    """Fallback model statistics display when result_card is unavailable."""
    if st is None:
        return

    is_mle = getattr(result, "is_mle_model", False)
    is_panel = getattr(result, "model_type", "") == "panel"
    is_mixedlm = getattr(result, "model_type", "") == "mixedlm"
    is_negbin = getattr(result, "model_type", "") == "negbin"

    # Row 1: Model fit metrics (differs by model type)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if is_mle:
            pr2 = getattr(result, "pseudo_r_squared", None)
            st.metric("伪 R² (McFadden)", f"{pr2:.4f}" if pr2 is not None else "N/A")
        elif is_panel:
            wr2 = getattr(result, "within_r_squared", None)
            st.metric("Within R²", f"{wr2:.4f}" if wr2 is not None else "N/A")
        else:
            r2 = getattr(result, "r_squared", None)
            st.metric("R²", f"{r2:.4f}" if r2 is not None else "N/A")
    with col2:
        if is_mle:
            llr_val = getattr(result, "llr", None)
            st.metric("似然比检验 (LR χ²)", f"{llr_val:.4f}" if llr_val is not None else "N/A")
        elif is_panel:
            br2 = getattr(result, "between_r_squared", None)
            st.metric("Between R²", f"{br2:.4f}" if br2 is not None else "N/A")
        elif is_mixedlm:
            gc = getattr(result, "group_count", None)
            st.metric("Groups", gc if gc else "N/A")
        elif is_negbin:
            disp = getattr(result, "dispersion", None)
            st.metric("Dispersion", f"{disp:.4f}" if disp is not None else "N/A")
        else:
            adj = getattr(result, "adj_r_squared", None)
            st.metric("Adj. R²", f"{adj:.4f}" if adj is not None else "N/A")
    with col3:
        if is_mle:
            llr_p = getattr(result, "llr_pvalue", None)
            st.metric("LR p值", f"{llr_p:.6f}" if llr_p is not None else "N/A")
        elif is_panel:
            or2 = getattr(result, "overall_r_squared", None)
            st.metric("Overall R²", f"{or2:.4f}" if or2 is not None else "N/A")
        elif is_mixedlm:
            re_var = getattr(result, "re_var", None)
            if re_var:
                re_str = ", ".join(f"{k}={v:.4f}" for k, v in re_var.items())
                st.metric("RE Var", re_str)
            else:
                st.metric("RE Var", "N/A")
        else:
            f_val = None
            f_stat = getattr(result, "f_statistic", None)
            if f_stat is not None:
                f_val = f_stat[0]
            st.metric("F 统计量", f"{f_val:.4f}" if f_val is not None else "N/A")
    with col4:
        if is_panel:
            ec = getattr(result, "entity_count", None)
            tc = getattr(result, "time_count", None)
            st.metric("Entities x Time", f"{ec} x {tc}" if ec and tc else "N/A")
        elif is_mixedlm:
            n = getattr(result, "n_obs", None)
            st.metric("N", n if n else "N/A")
        else:
            n = getattr(result, "n_obs", None)
            st.metric("N", n if n else "N/A")

    # Row 2: Information criteria and LL (same for both model types)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        aic = getattr(result, "aic", None)
        st.metric("AIC", f"{aic:.2f}" if aic else "N/A")
    with col2:
        bic = getattr(result, "bic", None)
        st.metric("BIC", f"{bic:.2f}" if bic else "N/A")
    with col3:
        ll = getattr(result, "log_likelihood", None)
        st.metric("Log-Likelihood", f"{ll:.4f}" if ll is not None else "N/A")
    with col4:
        if is_panel:
            pt = getattr(result, "panel_type", None)
            st.metric("Estimator", pt if pt else "Panel")
        elif is_mixedlm:
            rmse = getattr(result, "rmse", None)
            st.metric("RMSE", f"{rmse:.4f}" if rmse else "N/A")
        elif not is_mle:
            rmse = getattr(result, "rmse", None)
            st.metric("RMSE", f"{rmse:.4f}" if rmse else "N/A")
        else:
            st.metric("Pseudo R²", f"{getattr(result, 'pseudo_r_squared', 0):.4f}" if getattr(result, 'pseudo_r_squared', None) is not None else "N/A")  # noqa: E501


def _fallback_coefficient_table(result: Any) -> None:
    """Fallback coefficient table when result_card is unavailable."""
    if st is None:
        return

    coefficients = getattr(result, "coefficients", None)
    if not coefficients:
        try:
            summary = (
                result.summary()
                if hasattr(result, "summary")
                else str(result)
            )
            st.text(summary)
        except Exception:
            st.info("系数表数据不可用。")
        return

    is_mle = getattr(result, "is_mle_model", False)
    model_type = getattr(result, "model_type", "")
    is_logit = model_type == "logit"
    is_count = getattr(result, "is_count_model", False)
    stat_label = "z 值" if is_mle else "t 值"

    table_data = []
    for coef in coefficients:
        name = getattr(coef, "name", "?")
        est = getattr(coef, "coef", None)
        se = getattr(coef, "se", None)
        t_stat = getattr(coef, "t_stat", None)
        p_val = getattr(coef, "pvalue", None)
        ci_lower = getattr(coef, "ci_lower", None)
        ci_upper = getattr(coef, "ci_upper", None)

        sig = ""
        if p_val is not None:
            if p_val <= 0.01:
                sig = "***"
            elif p_val <= 0.05:
                sig = "**"
            elif p_val <= 0.1:
                sig = "*"

        row = {
            "变量": name,
            "系数": f"{est:.4f}" if est is not None else "-",
            "标准误": f"{se:.4f}" if se is not None else "-",
            stat_label: f"{t_stat:.4f}" if t_stat is not None else "-",
            "p 值": f"{p_val:.4f}" if p_val is not None else "-",
            "显著性": sig,
            "95% CI": (
                f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                if ci_lower is not None and ci_upper is not None
                else "-"
            ),
        }
        # Only logit (not probit) gets odds ratio — probit is on probit scale
        if is_logit and est is not None:
            import math
            or_val = math.exp(est)
            row["OR (几率比)"] = f"{or_val:.4f}"
        # Count models get Incidence Rate Ratio
        if is_count and est is not None:
            import math
            irr_val = math.exp(est)
            row["IRR (exp(B))"] = f"{irr_val:.4f}"
        table_data.append(row)

    if table_data:
        st.dataframe(
            table_data,
            use_container_width=True,
        )
        st.caption("显著性标记: *** p < 0.01, ** p < 0.05, * p < 0.1")


# =========================================================================
# Diagnostic plot helpers (Phase 1)
# =========================================================================


def _render_scatter_plot(result: Any, df: Any, model_spec: Any) -> None:
    """渲染散点图 + 回归线。"""
    if st is None or not VIS_AVAILABLE:
        return

    if model_spec is None or df is None:
        st.info("需要模型规格和数据才能显示散点图。")
        return

    dep_var = model_spec.dep_var
    indep_vars = model_spec.indep_vars

    if len(indep_vars) == 1:
        try:
            fig = scatter_with_regression(
                df,
                x_col=indep_vars[0],
                y_col=dep_var,
                title=f"{dep_var} vs {indep_vars[0]}（含回归线）",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"散点图渲染失败: {e}")
    else:
        st.info(f"有 {len(indep_vars)} 个自变量。无法在单个散点图中展示多元回归关系。")

        selected_x = st.selectbox(
            "选择 X 轴变量（仅展示偏相关）",
            options=indep_vars,
            key="scatter_x",
        )
        try:
            fig = scatter_with_regression(
                df,
                x_col=selected_x,
                y_col=dep_var,
                title=f"{dep_var} vs {selected_x}（简单回归，非偏相关）",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"散点图渲染失败: {e}")


def _render_residual_plot(result: Any, df: Any) -> None:
    """渲染残差 vs 拟合值图。"""
    if st is None or not VIS_AVAILABLE:
        st.info("请安装可视化依赖以显示残差图。")
        return

    try:
        fig = residual_vs_fitted_plot(result, df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"残差图渲染失败: {e}")


def _render_qq_plot(result: Any) -> None:
    """渲染 Q-Q 图。"""
    if st is None or not VIS_AVAILABLE:
        st.info("请安装可视化依赖以显示 Q-Q 图。")
        return

    try:
        residuals = getattr(result, "residuals", None)
        if residuals is not None:
            fig = qq_plot(np.asarray(residuals).flatten())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("模型结果中无残差数据。")
    except Exception as e:
        st.error(f"Q-Q 图渲染失败: {e}")


# =========================================================================
# Phase 2: Additional diagnostic plots
# =========================================================================


def _render_diagnostic_dashboard(result: Any, df: Any) -> None:
    """渲染 2x2 诊断总览面板。"""
    if st is None or not VIS_AVAILABLE:
        st.info("诊断图模块未加载。")
        return

    try:
        figs = diagnostic_dashboard(result, df)

        cols = st.columns(2)
        fig_titles = [
            ("residual_fitted", "残差 vs 拟合值"),
            ("qq", "正态 Q-Q 图"),
            ("scale_location", "尺度-位置图"),
            ("cooks_distance", "Cook's Distance"),
        ]

        for i, (fig_key, title) in enumerate(fig_titles):
            with cols[i % 2]:
                fig = figs.get(fig_key)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"dash_{fig_key}")
                else:
                    st.info(f"{title} 不可用")
    except Exception as e:
        st.error(f"诊断总览渲染失败: {e}")


def _render_scale_location_plot(result: Any, df: Any) -> None:
    """渲染尺度-位置图。"""
    if st is None or not VIS_AVAILABLE:
        st.info("请安装可视化依赖以显示尺度-位置图。")
        return

    try:
        fig = scale_location_plot(result, df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"尺度-位置图渲染失败: {e}")


def _render_cooks_distance_plot(result: Any, df: Any) -> None:
    """渲染 Cook's distance 图。"""
    if st is None or not VIS_AVAILABLE:
        st.info("请安装可视化依赖以显示 Cook's distance 图。")
        return

    try:
        fig = cooks_distance_plot(result, df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Cook's distance 图渲染失败: {e}")


def _save_fig_to_session(result: Any, df: Any, fig_name: str) -> None:
    """将诊断图暂存到 session_state 供导出页面使用。"""
    if st is None:
        return
    try:
        fig_map = {
            "residual_vs_fitted": lambda: residual_vs_fitted_plot(result, df),
            "qq_plot": lambda: qq_plot(np.asarray(getattr(result, "residuals", [])).flatten())
            if getattr(result, "residuals", None) is not None
            else None,
            "scale_location": lambda: scale_location_plot(result, df),
            "cooks_distance": lambda: cooks_distance_plot(result, df),
        }
        fn = fig_map.get(fig_name)
        if fn is not None:
            fig = fn()
            if fig is not None:
                if "export_charts" not in st.session_state:
                    st.session_state.export_charts = {}
                st.session_state.export_charts[fig_name] = fig
    except Exception:
        pass


# 页面入口
render()

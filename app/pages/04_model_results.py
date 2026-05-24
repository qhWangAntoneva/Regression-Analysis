# encoding: utf-8
"""
回归结果展示页面

Streamlit 页面：系数表、模型统计量、诊断图。
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]

VIS_AVAILABLE = False
try:
    from src.visualization.scatter import scatter_with_regression
    from src.visualization.residual import residual_vs_fitted_plot, qq_plot

    VIS_AVAILABLE = True
except ImportError:
    pass


def render() -> None:
    """渲染回归结果页面。"""
    if st is None:
        return

    st.title(":material/bar_chart: 回归结果")

    # 检查模型结果是否存在
    model_result = st.session_state.get("model_result")
    if model_result is None:
        st.warning("请先在「模型设定」页面运行回归模型。")
        st.page_link("app/pages/03_model_spec.py", label="前往模型设定", icon="⚙️")
        return

    df = st.session_state.get("data")
    variables = st.session_state.get("variables")
    model_spec = st.session_state.get("model_spec")

    # 模型摘要
    st.subheader("模型摘要")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        r2 = getattr(model_result, "rsquared", None)
        st.metric("R²", f"{r2:.4f}" if r2 is not None else "N/A")

    with col2:
        r2_adj = getattr(model_result, "rsquared_adj", None)
        st.metric("Adj. R²", f"{r2_adj:.4f}" if r2_adj is not None else "N/A")

    with col3:
        f_stat = getattr(model_result, "f_statistic", None)
        f_pvalue = getattr(model_result, "f_pvalue", None)
        f_label = f"{f_stat:.4f}" if f_stat is not None else "N/A"
        if f_pvalue is not None:
            f_label += f"\n(p={f_pvalue:.4f})"
        st.metric("F 统计量", f_label)

    with col4:
        nobs = getattr(model_result, "nobs", None)
        st.metric("样本量 (N)", nobs if nobs is not None else "N/A")

    # 第二行模型统计量
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        aic = getattr(model_result, "aic", None)
        st.metric("AIC", f"{aic:.2f}" if aic is not None else "N/A")

    with col2:
        bic = getattr(model_result, "bic", None)
        st.metric("BIC", f"{bic:.2f}" if bic is not None else "N/A")

    with col3:
        loglik = getattr(model_result, "log_likelihood", None)
        st.metric("Log-Likelihood", f"{loglik:.4f}" if loglik is not None else "N/A")

    with col4:
        rmse = getattr(model_result, "rmse", None)
        st.metric("RMSE", f"{rmse:.4f}" if rmse is not None else "N/A")

    # 系数表
    st.divider()
    st.subheader("系数表")

    coefficients = getattr(model_result, "coefficients", None)
    if coefficients:
        _render_coefficient_table(coefficients)
    else:
        # 尝试从 model_result 的 summary 属性获取
        try:
            summary = model_result.summary() if hasattr(model_result, "summary") else str(model_result)
            st.text(summary)
        except Exception:
            st.info("系数表数据不可用。")

    # 模型公式
    if model_spec:
        formula = f"{model_spec.dependent_var} ~ {' + '.join(model_spec.independent_vars)}"
        with st.expander("模型公式", expanded=False):
            st.code(formula, language="text")

    # 诊断图
    st.divider()
    st.subheader("诊断图")

    if not PLOTLY_AVAILABLE or not VIS_AVAILABLE:
        st.warning("请安装 plotly 和 statsmodels 以显示诊断图。")
        st.code("pip install plotly statsmodels", language="bash")
        return

    tab1, tab2, tab3 = st.tabs(["散点图 + 回归线", "残差 vs 拟合值", "Q-Q 图"])

    with tab1:
        _render_scatter_plot(model_result, df, model_spec)

    with tab2:
        _render_residual_plot(model_result, df)

    with tab3:
        _render_qq_plot(model_result)


def _render_coefficient_table(coefficients) -> None:
    """渲染格式化的系数表。"""
    if st is None:
        return

    table_data = []
    for coef in coefficients:
        name = getattr(coef, "name", getattr(coef, "var_name", "?"))
        est = getattr(coef, "coefficient", getattr(coef, "estimate", getattr(coef, "coef", None)))
        se = getattr(coef, "std_err", getattr(coef, "se", None))
        t_stat = getattr(coef, "t_statistic", getattr(coef, "t_value", getattr(coef, "t", None)))
        p_val = getattr(coef, "p_value", getattr(coef, "p", None))
        ci_lower = getattr(coef, "ci_lower", None)
        ci_upper = getattr(coef, "ci_upper", None)

        # 显著性星标
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
            "t 值": f"{t_stat:.4f}" if t_stat is not None else "-",
            "p 值": f"{p_val:.4f}" if p_val is not None else "-",
            "显著性": sig,
            "95% CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]" if ci_lower is not None and ci_upper is not None else "-",
        }
        table_data.append(row)

    if table_data:
        # 高亮显著系数
        def _highlight_sig(row_dict: dict) -> str:
            p_str = row_dict.get("p 值", "-")
            if p_str != "-":
                try:
                    p = float(p_str)
                    if p <= 0.05:
                        return "background-color: #e8f5e9"  # light green
                except ValueError:
                    pass
            return ""

        st.dataframe(
            table_data,
            use_container_width=True,
            column_config={
                "变量": st.column_config.TextColumn("变量"),
                "系数": st.column_config.TextColumn("系数"),
                "标准误": st.column_config.TextColumn("标准误"),
                "t 值": st.column_config.TextColumn("t 值"),
                "p 值": st.column_config.TextColumn("p 值"),
                "显著性": st.column_config.TextColumn("显著性"),
                "95% CI": st.column_config.TextColumn("95% 置信区间"),
            },
        )

        # 显著性注释
        st.caption("显著性标记: *** p < 0.01, ** p < 0.05, * p < 0.1")


def _render_scatter_plot(model_result, df, model_spec) -> None:
    """渲染散点图 + 回归线。"""
    if st is None or not VIS_AVAILABLE:
        return

    if model_spec is None or df is None:
        st.info("需要模型规格和数据才能显示散点图。")
        return

    dep_var = model_spec.dependent_var
    indep_vars = model_spec.independent_vars

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

        # 允许用户选择特定自变量绘图
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


def _render_residual_plot(model_result, df) -> None:
    """渲染残差 vs 拟合值图。"""
    if st is None or not VIS_AVAILABLE:
        st.info("请安装可视化依赖以显示残差图。")
        return

    try:
        fig = residual_vs_fitted_plot(model_result, df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"残差图渲染失败: {e}")


def _render_qq_plot(model_result) -> None:
    """渲染 Q-Q 图。"""
    if st is None or not VIS_AVAILABLE:
        st.info("请安装可视化依赖以显示 Q-Q 图。")
        return

    try:
        residuals = getattr(model_result, "residuals", None)
        if residuals is not None:
            import numpy as np

            fig = qq_plot(np.asarray(residuals).flatten())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("模型结果中无残差数据。")
    except Exception as e:
        st.error(f"Q-Q 图渲染失败: {e}")


# 页面入口
render()

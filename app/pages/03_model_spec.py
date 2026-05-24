# encoding: utf-8
"""
模型设定页面

Streamlit 页面：变量选择、模型规格、运行回归。
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports — avoid top-level crashes when modules not installed
MODELING_AVAILABLE = False
try:
    from app.components.variable_selector import render_variable_selector
    from src.modeling.specification import ModelSpec
    from src.modeling.fitter import run_ols
    from src.results.table import ModelResult

    MODELING_AVAILABLE = True
except ImportError:
    pass


def render() -> None:
    """渲染模型规格页面。"""
    if st is None:
        return

    st.title(":material/tune: 模型设定")

    # 检查数据是否存在
    if st.session_state.get("data") is None:
        st.warning("请先在「数据上传」页面上传数据集。")
        st.page_link("app/pages/01_data_upload.py", label="前往数据上传", icon="📂")
        return

    if not MODELING_AVAILABLE:
        st.warning("统计引擎模块未完全加载。请先安装依赖: `pip install statsmodels pandas`")
        # 即使没有 modeling 模块，也显示变量选择器（UI 预览）
        from src.preprocessing.type_detector import VariableTypeDetector

        variables = st.session_state.get("variables")
        if variables is None:
            detector = VariableTypeDetector()
            variables = detector.detect(st.session_state.data)

        from app.components.variable_selector import render_variable_selector

        dep_var, indep_vars = render_variable_selector(variables)
        if dep_var and indep_vars:
            st.info("选择已完成。请安装 statsmodels 以运行回归。")
        return

    df = st.session_state.data
    variables = st.session_state.variables

    # 确保 variables 存在
    if variables is None:
        from src.preprocessing.type_detector import VariableTypeDetector

        detector = VariableTypeDetector()
        variables = detector.detect(df)
        st.session_state.variables = variables

    # 变量选择器
    st.subheader("变量选择")

    dep_var, indep_vars = render_variable_selector(variables)

    # 模型选项
    st.divider()
    st.subheader("模型选项")

    col1, col2, col3 = st.columns(3)

    with col1:
        add_constant = st.checkbox("包含常数项 (截距)", value=True, help="OLS 回归默认包含截距项")

    with col2:
        ci_level = st.select_slider(
            "置信区间水平",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            help="系数置信区间的置信水平",
        )

    with col3:
        use_robust = st.checkbox("稳健标准误 (HC3)", value=False, help="异方差稳健标准误")

    # 显示所选变量公式
    if dep_var and indep_vars:
        formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
        if add_constant:
            formula += " + 常数项"
        st.code(f"模型公式: {formula}", language="text")

    # 运行回归按钮
    st.divider()

    run_disabled = not (dep_var and len(indep_vars) >= 1)

    if st.button(
        ":material/play_arrow: 运行回归",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    ):
        _run_regression(df, dep_var, indep_vars, add_constant, ci_level, use_robust)


def _run_regression(
    df,
    dep_var: str,
    indep_vars: list[str],
    add_constant: bool,
    ci_level: float,
    use_robust: bool,
) -> None:
    """运行回归并保存结果。"""
    if st is None:
        return

    with st.spinner("正在运行回归模型..."):
        try:
            # 构建 ModelSpec
            spec = ModelSpec(
                dependent_var=dep_var,
                independent_vars=indep_vars,
                add_constant=add_constant,
                ci_level=ci_level,
                use_robust_se=use_robust,
            )

            # 运行 OLS
            result: ModelResult = run_ols(df, spec)

            # 保存到 session state
            st.session_state.model_result = result
            st.session_state.model_spec = spec
            st.session_state.model_run_time = True

            st.success("回归模型运行成功！")
            st.page_link("app/pages/04_model_results.py", label="查看回归结果", icon="📊")

            # 显示简要结果
            if hasattr(result, "rsquared"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R²", f"{result.rsquared:.4f}")
                with col2:
                    st.metric("Adj. R²", f"{result.rsquared_adj:.4f}")
                with col3:
                    st.metric("N", result.nobs)

        except Exception as e:
            st.error(f"回归模型运行失败: {e}")


# 页面入口
render()

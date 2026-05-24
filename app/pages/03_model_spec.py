# encoding: utf-8
"""
模型设定页面

Streamlit 页面：变量选择、模型规格、运行回归，支持多模型对比。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports — avoid top-level crashes when modules not installed
MODELING_AVAILABLE = False
try:
    from app.components.variable_selector import render_variable_selector
    from app.components.model_control import (
        render_model_comparison_controls,
        render_model_controls,
    )
    from src.modeling.fitter import ModelFitter
    from src.modeling.specification import ModelSpec, build_formula
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

    # 保存可用变量到 session state（供模型对比组件使用）
    if variables:
        st.session_state["available_vars"] = [v.name for v in variables]
    if dep_var:
        st.session_state["selected_dep_var"] = dep_var

    # ------------------------------------------------------------------
    # Phase 2: 高级选项 (model control panel)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("模型选项")

    model_config = render_model_controls(key_prefix="model_main")

    # 公式预览
    if dep_var and indep_vars:
        try:
            spec_preview = ModelSpec(
                dep_var=dep_var,
                indep_vars=indep_vars,
                has_intercept=model_config["add_constant"],
            )
            formula_str = build_formula(spec_preview)
            st.code(f"模型公式: {formula_str}", language="text")
        except Exception:
            # Fallback manual formula display
            formula_str = f"{dep_var} ~ {' + '.join(indep_vars)}"
            if not model_config["add_constant"]:
                formula_str += " - 1"
            st.code(f"模型公式: {formula_str}", language="text")

    # ------------------------------------------------------------------
    # Phase 2: 多模型对比
    # ------------------------------------------------------------------
    comparison_controls = render_model_comparison_controls(key_prefix="compare")
    comparison_specs = comparison_controls.get("comparison_specs", [])
    compare_mode = len(comparison_specs) > 0

    # ------------------------------------------------------------------
    # 运行按钮区域
    # ------------------------------------------------------------------
    st.divider()

    run_disabled = not (dep_var and len(indep_vars) >= 1)

    col_run, col_compare = st.columns([1, 1])

    with col_run:
        if st.button(
            ":material/play_arrow: 运行回归",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        ):
            _run_regression(df, dep_var, indep_vars, model_config)

    with col_compare:
        compare_disabled = run_disabled or not compare_mode
        if st.button(
            ":material/compare_arrows: 运行所有模型",
            type="secondary",
            use_container_width=True,
            disabled=compare_disabled,
        ):
            _run_all_models(
                df, dep_var, indep_vars, model_config, comparison_specs
            )


def _run_regression(
    df: Any,
    dep_var: str,
    indep_vars: List[str],
    model_config: Dict[str, Any],
) -> None:
    """运行单个回归模型并保存结果到 session state。"""
    if st is None:
        return

    with st.spinner("正在运行回归模型..."):
        try:
            spec = ModelSpec(
                dep_var=dep_var,
                indep_vars=indep_vars,
                has_intercept=model_config["add_constant"],
            )

            result: ModelResult = run_ols(df, spec)

            # 保存到 session state
            st.session_state.model_result = result
            st.session_state.model_spec = spec
            st.session_state.model_run_time = True
            st.session_state.model_config = model_config

            # Clear multi-model state
            st.session_state.model_results_list = [result]

            st.success("回归模型运行成功！")
            st.page_link(
                "app/pages/04_model_results.py",
                label="查看回归结果",
                icon="📊",
            )

            # 显示简要结果
            _display_quick_summary(result)

        except Exception as e:
            st.error(f"回归模型运行失败: {e}")


def _run_all_models(
    df: Any,
    dep_var: str,
    indep_vars: List[str],
    model_config: Dict[str, Any],
    comparison_specs: List[ModelSpec],
) -> None:
    """运行主模型及所有对比模型，结果存入 session_state。"""
    if st is None:
        return

    with st.spinner("正在运行所有模型..."):
        try:
            fitter = ModelFitter()
            results: List[ModelResult] = []

            # 主模型
            main_spec = ModelSpec(
                dep_var=dep_var,
                indep_vars=indep_vars,
                has_intercept=model_config["add_constant"],
            )
            main_result = fitter.fit(main_spec, df)
            results.append(main_result)

            # 对比模型
            for comp_spec in comparison_specs:
                comp_result = fitter.fit(comp_spec, df)
                results.append(comp_result)

            # 保存到 session state
            st.session_state.model_result = results[0]
            st.session_state.model_spec = main_spec
            st.session_state.model_results_list = results
            st.session_state.model_run_time = True
            st.session_state.model_config = model_config

            st.success(f"共运行 {len(results)} 个模型（1个主模型 + {len(results)-1}个对比模型）！")
            st.page_link(
                "app/pages/04_model_results.py",
                label="查看回归结果",
                icon="📊",
            )

            # 显示简要结果
            for i, res in enumerate(results):
                label = "主模型" if i == 0 else f"对比模型 {i}"
                with st.expander(f"{label}: {res.specification}", expanded=(i == 0)):
                    _display_quick_summary(res)

        except Exception as e:
            st.error(f"模型运行失败: {e}")


def _display_quick_summary(result: ModelResult) -> None:
    """显示模型的快速摘要（R², Adj-R², N）。"""
    if st is None:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        r2 = f"{result.r_squared:.4f}" if result.r_squared is not None else "N/A"
        st.metric("R²", r2)
    with col2:
        adj = f"{result.adj_r_squared:.4f}" if result.adj_r_squared is not None else "N/A"
        st.metric("Adj. R²", adj)
    with col3:
        st.metric("N", result.n_obs)


# 页面入口
render()

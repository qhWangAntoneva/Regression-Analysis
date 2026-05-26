# encoding: utf-8
"""
模型设定页面

Streamlit 页面：变量选择、模型规格、运行回归，支持多模型对比。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports — avoid top-level crashes when modules not installed
MODELING_AVAILABLE = False
try:
    from app.components.variable_selector import (
        render_interaction_ui,
        render_transforms_ui,
        render_variable_selector,
    )
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


def _render_data_filter(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """渲染数据筛选（行过滤）界面并返回过滤后的数据。

    数值列提供范围滑块，分类列提供多选，二值列提供值选择。
    筛选结果存入 st.session_state.filtered_data。

    Args:
        df: 原始 DataFrame。
        variables: VariableInfo 列表。

    Returns:
        过滤后的 DataFrame（若无筛选则返回原始 df）。
    """
    if st is None:
        return df

    # 检查是否已有筛选
    filtered = st.session_state.get("filtered_data")
    if filtered is not None:
        current_df = filtered
    else:
        current_df = df

    n_total = len(df)
    n_current = len(current_df)

    with st.expander(
        f":material/filter_alt: 数据筛选"
        f"（当前: {n_current}/{n_total} 行）",
        expanded=filtered is not None,
    ):
        if variables is None:
            st.info("数据不可用。")
            return current_df

        # 分离列类型
        numeric_vars = [v for v in variables if v.inferred_type in ("continuous",)]
        cat_vars = [v for v in variables if v.inferred_type in ("categorical", "binary", "ordinal")]

        filters_applied = []
        col_left, col_right = st.columns(2)

        # 数值列滑块
        with col_left:
            st.markdown("**数值列范围**")
            if not numeric_vars:
                st.info("无连续数值列。")
            else:
                for v in numeric_vars[:8]:  # 最多显示 8 个滑块
                    col_min = float(df[v.name].min())
                    col_max = float(df[v.name].max())
                    # 使用 session_state 记忆滑块值
                    key_min = f"filter_min_{v.name}"
                    key_max = f"filter_max_{v.name}"
                    default_min = st.session_state.get(key_min, col_min)
                    default_max = st.session_state.get(key_max, col_max)

                    vals = st.slider(
                        v.name,
                        min_value=col_min,
                        max_value=col_max,
                        value=(default_min, default_max),
                        key=f"filter_slider_{v.name}",
                    )
                    st.session_state[key_min] = vals[0]
                    st.session_state[key_max] = vals[1]
                    if vals[0] > col_min or vals[1] < col_max:
                        filters_applied.append((v.name, vals))

        # 分类列多选
        with col_right:
            st.markdown("**分类列筛选**")
            if not cat_vars:
                st.info("无分类变量。")
            else:
                for v in cat_vars[:6]:  # 最多显示 6 个
                    unique_vals = sorted(df[v.name].dropna().unique().tolist())
                    selected = st.multiselect(
                        v.name,
                        options=unique_vals,
                        default=st.session_state.get(f"filter_cat_{v.name}", unique_vals),
                        key=f"filter_cat_{v.name}",
                    )
                    st.session_state[f"filter_cat_{v.name}"] = selected
                    if selected and set(selected) != set(unique_vals):
                        filters_applied.append((v.name, selected))

        # 筛选按钮
        st.divider()
        col_apply, col_clear, _ = st.columns([1, 1, 2])
        with col_apply:
            if st.button(
                ":material/check: 应用筛选",
                type="primary",
                use_container_width=True,
                key="apply_filters",
            ):
                filtered_df = df.copy()
                # 应用数值列范围筛选
                for name, vals in filters_applied:
                    if isinstance(vals, tuple):
                        filtered_df = filtered_df[
                            (filtered_df[name] >= vals[0]) & (filtered_df[name] <= vals[1])
                        ]
                    elif isinstance(vals, list):
                        filtered_df = filtered_df[filtered_df[name].isin(vals)]

                st.session_state.filtered_data = filtered_df
                st.success(f"筛选后: {len(filtered_df)}/{len(df)} 行")
                st.rerun()

        with col_clear:
            if st.button(
                ":material/clear: 清除筛选",
                use_container_width=True,
                key="clear_filters",
            ):
                # 清除所有筛选状态
                if "filtered_data" in st.session_state:
                    del st.session_state.filtered_data
                for key in list(st.session_state.keys()):
                    if key.startswith("filter_min_") or key.startswith("filter_max_") or key.startswith("filter_cat_"):
                        del st.session_state[key]
                st.rerun()

    return current_df


def render() -> None:
    """渲染模型规格页面。"""
    if st is None:
        return

    st.title(":material/tune: 模型设定")

    # Gallery mode notice
    if st.session_state.get("gallery_mode"):
        st.info(
            ":material/info: 当前使用示例数据。变量已自动填充（预计算模型的设定）。"
            "您可以修改设定后重新运行回归，对比不同模型的结果。"
        )

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

    # 数据筛选
    df = _render_data_filter(df, variables)

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
    # Phase 3.1: 变量转换 UI
    # ------------------------------------------------------------------
    transforms: Dict[str, str] = {}
    interaction_terms: List[Tuple[str, str]] = []
    if dep_var and indep_vars:
        with st.expander("变量转换", expanded=False):
            transforms = render_transforms_ui(
                indep_vars, variables, key_prefix="var"
            )
            st.divider()
            interaction_terms = render_interaction_ui(
                indep_vars, key_prefix="var"
            )

    # ------------------------------------------------------------------
    # Phase 2: 高级选项 (model control panel)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("模型选项")

    model_config = render_model_controls(key_prefix="model_main")

    # ------------------------------------------------------------------
    # Auto-suggest and validate model type based on dependent variable
    # ------------------------------------------------------------------
    if dep_var and df is not None:
        dep_series = df[dep_var].dropna()
        unique_vals = dep_series.nunique()
        is_binary = unique_vals == 2

        if model_config["model_type"] in ("logit", "probit") and not is_binary:
            st.warning(
                f":material/warning: 因变量「{dep_var}」有 {unique_vals} 个不同值，"
                f"不适用于 Logit 模型。Logit 要求二分类因变量 (0/1)。"
                f"请切换为 OLS 或选择二分类因变量。"
            )
        elif model_config["model_type"] == "ols" and is_binary:
            st.info(
                f":material/info: 因变量「{dep_var}」只有 2 个不同值 ({sorted(dep_series.unique().tolist())})，"
                f"可能是二分类变量。建议使用「Logit」模型进行逻辑回归。"
            )

    # 公式预览（含转换和交互项）
    if dep_var and indep_vars:
        try:
            spec_preview = ModelSpec(
                dep_var=dep_var,
                indep_vars=indep_vars,
                has_intercept=model_config["add_constant"],
                transforms=transforms,
                interaction_terms=interaction_terms,
                model_type=model_config["model_type"],
            )
            formula_str = build_formula(spec_preview)
            st.code(f"模型公式: {formula_str}", language="text")
            if transforms:
                parts = [f"{t}({v})" for v, t in transforms.items()]
                st.caption("变量转换: " + ", ".join(parts))
            if interaction_terms:
                parts = [f"{v1}:{v2}" for v1, v2 in interaction_terms]
                st.caption("交互项: " + ", ".join(parts))
        except Exception:
            # Fallback manual formula display
            formula_str = f"{dep_var} ~ {' + '.join(indep_vars)}"
            if not model_config["add_constant"]:
                formula_str += " - 1"
            if transforms:
                formula_str += "  [转换: " + ", ".join(f"{t}({v})" for v, t in transforms.items()) + "]"
            if interaction_terms:
                formula_str += "  [交互: " + ", ".join(f"{v1}:{v2}" for v1, v2 in interaction_terms) + "]"
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
            _run_regression(
                df, dep_var, indep_vars, model_config,
                transforms, interaction_terms,
            )

    with col_compare:
        compare_disabled = run_disabled or not compare_mode
        if st.button(
            ":material/compare_arrows: 运行所有模型",
            type="secondary",
            use_container_width=True,
            disabled=compare_disabled,
        ):
            _run_all_models(
                df, dep_var, indep_vars, model_config,
                comparison_specs, transforms, interaction_terms,
            )


def _run_regression(
    df: Any,
    dep_var: str,
    indep_vars: List[str],
    model_config: Dict[str, Any],
    transforms: Dict[str, str],
    interaction_terms: List[Tuple[str, str]],
) -> None:
    """运行单个回归模型并保存结果到 session state。

    支持变量转换、交互项和稳健标准误。
    """
    if st is None:
        return

    with st.spinner("正在运行回归模型..."):
        try:
            spec = ModelSpec(
                dep_var=dep_var,
                indep_vars=indep_vars,
                has_intercept=model_config["add_constant"],
                transforms=transforms,
                interaction_terms=interaction_terms,
                missing_strategy=model_config.get("missing_handling", "drop"),
                model_type=model_config.get("model_type", "ols"),
            )

            cov_type = model_config.get("se_type", "nonrobust")
            fitter = ModelFitter()
            result: ModelResult = fitter.fit(df, spec, cov_type=cov_type)

            # 保存到 session state
            st.session_state.model_result = result
            st.session_state.model_spec = spec
            st.session_state.model_run_time = True
            st.session_state.model_config = model_config

            # Clear gallery mode when user runs their own model
            st.session_state.gallery_mode = False
            st.session_state.gallery_item_id = None
            st.session_state.gallery_item_title = None

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
    transforms: Dict[str, str],
    interaction_terms: List[Tuple[str, str]],
) -> None:
    """运行主模型及所有对比模型，结果存入 session_state。

    支持变量转换、交互项和稳健标准误。
    """
    if st is None:
        return

    with st.spinner("正在运行所有模型..."):
        try:
            cov_type = model_config.get("se_type", "nonrobust")
            fitter = ModelFitter()
            results: List[ModelResult] = []

            # 主模型
            main_spec = ModelSpec(
                dep_var=dep_var,
                indep_vars=indep_vars,
                has_intercept=model_config["add_constant"],
                transforms=transforms,
                interaction_terms=interaction_terms,
                missing_strategy=model_config.get("missing_handling", "drop"),
                model_type=model_config.get("model_type", "ols"),
            )
            main_result = fitter.fit(
                main_spec, df, cov_type=cov_type,
            )
            results.append(main_result)

            # 对比模型
            for comp_spec in comparison_specs:
                comp_result = fitter.fit(
                    comp_spec, df, cov_type=cov_type,
                )
                results.append(comp_result)

            # 保存到 session state
            st.session_state.model_result = results[0]
            st.session_state.model_spec = main_spec
            st.session_state.model_results_list = results
            st.session_state.model_run_time = True
            st.session_state.model_config = model_config

            # Clear gallery mode when user runs their own model
            st.session_state.gallery_mode = False
            st.session_state.gallery_item_id = None
            st.session_state.gallery_item_title = None

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
    """显示模型的快速摘要（R²/伪R², F/LR, N）。"""
    if st is None:
        return

    is_mle = getattr(result, "is_mle_model", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        if is_mle:
            pr2 = f"{result.pseudo_r_squared:.4f}" if result.pseudo_r_squared is not None else "N/A"
            st.metric("伪 R² (McFadden)", pr2)
        else:
            r2 = f"{result.r_squared:.4f}" if result.r_squared is not None else "N/A"
            st.metric("R²", r2)
    with col2:
        if is_mle:
            llr_val = f"{result.llr:.4f}" if result.llr is not None else "N/A"
            st.metric("LR χ²", llr_val)
        else:
            adj = f"{result.adj_r_squared:.4f}" if result.adj_r_squared is not None else "N/A"
            st.metric("Adj. R²", adj)
    with col3:
        st.metric("N", result.n_obs)


# 页面入口
render()

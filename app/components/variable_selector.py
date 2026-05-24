# encoding: utf-8
"""
变量选择器 UI 组件

提供因变量和自变量的选择界面，以及：
- 变量转换（log、标准化、中心化、平方）
- 交互项创建
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.preprocessing.type_detector import VariableInfo

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


def render_variable_selector(
    variables: list[VariableInfo],
    key_prefix: str = "var",
) -> tuple[str | None, list[str]]:
    """变量选择器。

    使用 st.selectbox 选择因变量，st.multiselect 选择自变量。
    自动过滤排除 id 类型变量。

    Args:
        variables: VariableInfo 对象列表。
        key_prefix: Streamlit 组件 key 前缀。

    Returns:
        (因变量名, 自变量名列表) 元组。如果用户未选择，因变量为 None。
    """
    if st is None:
        return None, []

    # 分类变量标记
    type_icons = {
        "continuous": "",
        "categorical": " 🏷️",
        "binary": " 🎯",
        "ordinal": " 🔢",
        "id": " 🆔",
        "text": " 📝",
    }

    # 过滤掉 id 类型和 text 类型变量（不适合建模）
    valid_vars = [v for v in variables if v.inferred_type not in ("id", "text")]
    id_vars = [v for v in variables if v.inferred_type == "id"]

    # 构建显示名称
    def display_name(v: VariableInfo) -> str:
        icon = type_icons.get(v.inferred_type, "")
        return f"{v.name}{icon}"

    # 生成选项列表
    var_options = {display_name(v): v.name for v in valid_vars}
    var_names_sorted = sorted(var_options.keys())

    col1, col2 = st.columns([1, 2])

    with col1:
        # 因变量选择
        dep_var_display = st.selectbox(
            "因变量 (Dependent Variable)",
            options=var_names_sorted,
            index=None,
            placeholder="请选择因变量...",
            key=f"{key_prefix}_dep",
            help="选择回归模型中的因变量（被解释变量）",
        )
        dep_var = var_options.get(dep_var_display) if dep_var_display else None

    with col2:
        # 自变量多选
        indep_vars_display = st.multiselect(
            "自变量 (Independent Variables)",
            options=var_names_sorted,
            default=None,
            key=f"{key_prefix}_indep",
            help="选择一个或多个自变量（解释变量）",
        )
        indep_vars = [var_options.get(d) for d in indep_vars_display if d in var_options]

    # 显示被排除的变量
    if id_vars:
        excluded_names = ", ".join(v.name for v in id_vars)
        st.caption(f"已自动排除 ID 列: {excluded_names}")

    # 变量统计摘要
    if dep_var:
        dep_info = next((v for v in variables if v.name == dep_var), None)
        if dep_info:
            with st.expander("因变量统计", expanded=False):
                st.json(
                    {
                        "名称": dep_info.name,
                        "推断类型": dep_info.inferred_type,
                        "唯一值": dep_info.n_unique,
                        "缺失值": f"{dep_info.n_missing} ({dep_info.missing_rate * 100:.1f}%)",
                        "均值": dep_info.mean,
                        "标准差": dep_info.std,
                        "范围": f"[{dep_info.min_val}, {dep_info.max_val}]" if dep_info.min_val is not None else "-",
                    }
                )

    return dep_var, indep_vars


def render_transforms_ui(
    indep_vars: list[str],
    variables: list[VariableInfo],
    key_prefix: str = "var",
) -> dict[str, str]:
    """变量转换 UI。

    为每个选中的连续自变量提供转换复选框。

    Args:
        indep_vars: 已选中的自变量列表。
        variables: VariableInfo 对象列表。
        key_prefix: Streamlit 组件 key 前缀。

    Returns:
        转换映射字典：{变量名: 转换类型}。
    """
    if st is None or not indep_vars:
        return {}

    var_info_map = {v.name: v for v in variables}
    transforms: dict[str, str] = {}

    # 只对连续变量显示转换选项
    continuous_selected = [
        v for v in indep_vars
        if v in var_info_map and var_info_map[v].inferred_type == "continuous"
    ]

    if not continuous_selected:
        return {}

    st.markdown("**变量转换**")
    st.caption("对选中的连续变量应用转换，转换后的变量将自动加入模型。")

    for var in continuous_selected:
        cols = st.columns([1, 3])
        with cols[0]:
            apply = st.checkbox(
                f"{var}",
                key=f"{key_prefix}_transform_{var}_enable",
                help=f"对 {var} 应用转换",
            )
        if apply:
            with cols[1]:
                ttype = st.selectbox(
                    "转换类型",
                    options=["log", "standardize", "center", "square"],
                    index=0,
                    key=f"{key_prefix}_transform_{var}_type",
                    help=(
                        "log: ln(x+1e-10), "
                        "standardize: Z-score, "
                        "center: 减去均值, "
                        "square: 平方"
                    ),
                    label_visibility="collapsed",
                )
                transforms[var] = ttype

    return transforms


def render_interaction_ui(
    indep_vars: list[str],
    key_prefix: str = "var",
) -> list[tuple[str, str]]:
    """交互项创建 UI。

    提供两个下拉框选择变量对，使用 patsy 的 ``:`` 语法创建交互项。
    已创建的交互项显示在列表中，每个项可独立删除。

    Args:
        indep_vars: 已选中的自变量列表。
        key_prefix: Streamlit 组件 key 前缀。

    Returns:
        当前所有交互项列表 ``[(var1, var2), ...]``。
    """
    if st is None or not indep_vars:
        return []

    key_inter = f"{key_prefix}_interaction_terms"
    if key_inter not in st.session_state:
        st.session_state[key_inter] = []

    st.markdown("**交互项**")
    st.caption("选择两个变量创建交互项（通过 patsy ``:`` 语法）。")

    interaction_terms = st.session_state[key_inter]

    if len(indep_vars) >= 2:
        col_v1, col_v2, col_btn = st.columns([2, 2, 1])
        with col_v1:
            v1 = st.selectbox(
                "变量 1",
                options=indep_vars,
                index=None,
                placeholder="选择...",
                key=f"{key_prefix}_inter_v1",
                label_visibility="collapsed",
            )
        with col_v2:
            v2 = st.selectbox(
                "变量 2",
                options=indep_vars,
                index=None,
                placeholder="选择...",
                key=f"{key_prefix}_inter_v2",
                label_visibility="collapsed",
            )
        with col_btn:
            if st.button(
                "添加",
                key=f"{key_prefix}_inter_add",
                use_container_width=True,
            ):
                if v1 and v2 and v1 != v2:
                    pair = (v1, v2)
                    if pair not in interaction_terms and (v2, v1) not in interaction_terms:
                        interaction_terms.append(pair)
                        st.rerun()
                else:
                    st.warning("请选择两个不同的变量。")

    # 显示已创建的交互项
    if interaction_terms:
        for idx, (v1, v2) in enumerate(interaction_terms):
            col_t, col_d = st.columns([5, 1])
            with col_t:
                st.text(f"  {v1}:{v2}")
            with col_d:
                if st.button(
                    "删除",
                    key=f"{key_prefix}_inter_del_{idx}",
                    use_container_width=True,
                ):
                    interaction_terms.pop(idx)
                    st.rerun()

    return list(interaction_terms)

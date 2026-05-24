# encoding: utf-8
"""
变量选择器 UI 组件

提供因变量和自变量的选择界面。
"""

from __future__ import annotations

from typing import Any

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

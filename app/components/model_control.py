# encoding: utf-8
"""Model control panel UI components.

Provides Streamlit widgets for configuring model parameters and
managing multi-model comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

from src.modeling.specification import ModelSpec


def render_model_controls(key_prefix: str = "model") -> Dict[str, Any]:
    """Render a model parameter configuration panel inside an expander.

    Wraps advanced model options in ``st.expander``.

    Args:
        key_prefix: Streamlit component key prefix for uniqueness.

    Returns:
        A dictionary of configuration values:
            - ``'model_type'``: str, ``'ols'`` or ``'logit'``.
            - ``'add_constant'``: bool, whether to include an intercept.
            - ``'ci_level'``: float, confidence interval level (0.90/0.95/0.99).
            - ``'se_type'``: str, either ``'classic'`` or ``'robust_hc1'``.
            - ``'missing_handling'``: str, ``'drop'`` / ``'mean'`` / ``'none'``.
    """
    if st is None:
        return {
            "model_type": "ols",
            "add_constant": True,
            "ci_level": 0.95,
            "se_type": "nonrobust",
            "missing_handling": "drop",
        }

    # --- Model type selector (always visible) ---
    model_type_label = st.selectbox(
        "模型类型",
        options=["OLS", "Logit"],
        index=0,
        key=f"{key_prefix}_model_type",
        help=(
            "OLS 适用于连续因变量，Logit 适用于二分类因变量 (0/1)。\n\n"
            "注意：Logit 模型不支持 F 检验，将使用似然比检验 (LR) 替代。"
        ),
    )
    is_mle = model_type_label == "Logit"
    # Future: add "Probit" to the dropdown above, then is_mle drives SE hiding.

    with st.expander("高级选项", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            add_constant = st.checkbox(
                "包含常数项(截距)",
                value=True,
                key=f"{key_prefix}_const",
                help="OLS 回归默认包含截距项。取消勾选则强制过原点回归。",
            )
        with col_b:
            ci_level = st.selectbox(
                "置信区间水平",
                options=[0.90, 0.95, 0.99],
                index=1,
                key=f"{key_prefix}_ci",
                help="系数置信区间的置信水平。",
            )

        # Standard error options — hidden for logit (MLE uses different theory)
        if not is_mle:
            se_type = st.radio(
                "标准误类型",
                options=[
                    "普通标准误",
                    "HC0 (异方差稳健)",
                    "HC1 (默认)",
                    "HC2",
                    "HC3",
                ],
                index=0,
                horizontal=True,
                key=f"{key_prefix}_se",
                help=(
                    "稳健标准误在异方差情况下提供更可靠的推断。\n"
                    "- HC0: 普通稳健标准误\n"
                    "- HC1: HC0 带小样本校正（Stata 默认）\n"
                    "- HC2/HC3: 更激进的校正"
                ),
            )
        else:
            se_type = "普通标准误"

        missing_handling = st.selectbox(
            "缺失值处理",
            options=["删除整行", "均值填充", "不处理"],
            index=0,
            key=f"{key_prefix}_missing",
            help="选择缺失值的处理方式。",
        )

    _se_map = {
        "普通标准误": "nonrobust",
        "HC0 (异方差稳健)": "HC0",
        "HC1 (默认)": "HC1",
        "HC2": "HC2",
        "HC3": "HC3",
    }
    return {
        "model_type": "logit" if is_mle else "ols",
        "add_constant": add_constant,
        "ci_level": ci_level,
        "se_type": _se_map.get(se_type, "nonrobust"),
        "missing_handling": "drop" if missing_handling == "删除整行" else ("mean" if missing_handling == "均值填充" else "none"),
    }


def render_model_comparison_controls(
    key_prefix: str = "compare",
) -> Dict[str, Any]:
    """Render multi-model comparison controls.

    Allows the user to add comparison models with different independent
    variable combinations, view them in a list, and remove unwanted ones.

    Args:
        key_prefix: Streamlit component key prefix for uniqueness.

    Returns:
        A dictionary:
            - ``'comparison_specs'``: list of ModelSpec objects for comparison.
            - ``'spec_summaries'``: list of summary strings for display.
    """
    if st is None:
        return {"comparison_specs": [], "spec_summaries": []}

    if f"{key_prefix}_specs" not in st.session_state:
        st.session_state[f"{key_prefix}_specs"] = []
    if f"{key_prefix}_summaries" not in st.session_state:
        st.session_state[f"{key_prefix}_summaries"] = []

    st.divider()
    st.subheader("多模型对比")

    # --- Add comparison model ---
    with st.expander("添加对比模型", expanded=False):
        comp_indep = st.multiselect(
            "自变量组合 (对比模型)",
            options=st.session_state.get("available_vars", []),
            default=None,
            key=f"{key_prefix}_comp_indep",
            help="选择不同于主模型的自变量组合以进行对比。",
        )

        if st.button(
            "添加对比模型",
            key=f"{key_prefix}_add_btn",
            type="secondary",
            use_container_width=True,
        ):
            dep_var = st.session_state.get("selected_dep_var")
            if dep_var and comp_indep:
                spec = ModelSpec(
                    dep_var=dep_var,
                    indep_vars=comp_indep,
                    has_intercept=st.session_state.get(
                        f"{key_prefix}_parent_const", True
                    ),
                )
                summary = f"{dep_var} ~ {' + '.join(comp_indep)}"
                st.session_state[f"{key_prefix}_specs"].append(spec)
                st.session_state[f"{key_prefix}_summaries"].append(summary)
                st.success(f"已添加对比模型: {summary}")
            else:
                st.warning("请先选择因变量和至少一个自变量。")

    # --- Display comparison model list ---
    specs = st.session_state.get(f"{key_prefix}_specs", [])
    summaries = st.session_state.get(f"{key_prefix}_summaries", [])

    if specs:
        st.caption(f"已添加 {len(specs)} 个对比模型:")
        for idx, (spec, summary) in enumerate(zip(specs, summaries)):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text(f"  {idx + 1}. {summary}")
            with col2:
                if st.button(
                    "删除",
                    key=f"{key_prefix}_del_{idx}",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state[f"{key_prefix}_specs"].pop(idx)
                    st.session_state[f"{key_prefix}_summaries"].pop(idx)
                    st.rerun()
    else:
        st.info("尚未添加对比模型。可在上方添加不同自变量组合的模型。")

    return {
        "comparison_specs": list(specs),
        "spec_summaries": list(summaries),
    }

# encoding: utf-8
"""
Streamlit 可复用数据表格组件

提供数据预览和变量信息表的渲染功能。
"""

from __future__ import annotations

from typing import Any

from src.preprocessing.type_detector import VariableInfo

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


def render_data_preview(df: Any, key: str = "data_preview") -> None:
    """使用 st.dataframe 显示可滚动数据表格。

    Args:
        df: pandas DataFrame。
        key: Streamlit 组件 key。
    """
    if st is None:
        return

    n_rows, n_cols = df.shape

    st.caption(f"共 {n_rows} 行 × {n_cols} 列 ｜ 仅显示前 100 行预览")

    # 构建列配置：显示列名和缺失值标记
    column_config: dict[str, Any] = {}
    for col in df.columns[:50]:  # 限制最多配置 50 列（Streamlit 限制）
        col_str = str(col)
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            column_config[col_str] = st.column_config.TextColumn(
                col_str,
                help=f"缺失值: {missing_count} / {n_rows} ({missing_count / max(n_rows, 1) * 100:.1f}%)",
            )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=400,
        column_config=column_config or None,
        key=key,
    )


def render_variable_info(variables: list[VariableInfo]) -> None:
    """显示变量信息表。

    Args:
        variables: VariableInfo 对象列表。
    """
    if st is None:
        return

    st.subheader("变量信息")

    # 类型图标映射
    type_icons = {
        "continuous": "📈",
        "categorical": "🏷️",
        "binary": "🎯",
        "ordinal": "🔢",
        "id": "🆔",
        "text": "📝",
    }

    # 构建表格数据
    table_data = []
    for v in variables:
        icon = type_icons.get(v.inferred_type, "❓")
        table_data.append(
            {
                "变量名": v.name,
                "类型": f"{icon} {v.inferred_type}",
                "Pandas dtype": v.dtype,
                "唯一值": v.n_unique,
                "缺失值": v.n_missing,
                "缺失率": f"{v.missing_rate * 100:.2f}%",
                "均值": f"{v.mean:.4f}" if v.mean is not None else "-",
                "标准差": f"{v.std:.4f}" if v.std is not None else "-",
                "最小值": f"{v.min_val:.4f}" if v.min_val is not None else "-",
                "最大值": f"{v.max_val:.4f}" if v.max_val is not None else "-",
            }
        )

    st.dataframe(
        table_data,
        use_container_width=True,
        height=min(400, 35 * (len(table_data) + 1)),
    )


def render_type_override_ui(variables: list[VariableInfo]) -> None:
    """渲染变量类型手动覆盖界面。

    在 st.expander 中显示变量列表，每行带类型下拉选择框。
    用户可手动覆盖变量的推断类型，点击「应用覆盖」后生效。

    Args:
        variables: VariableInfo 对象列表。
    """
    if st is None:
        return

    # 类型选项（排除 inferred_type 中不存在但语义清晰的类型）
    type_options = ["continuous", "categorical", "binary", "ordinal", "id", "text"]

    type_labels = {
        "continuous": "连续变量 (Continuous)",
        "categorical": "分类变量 (Categorical)",
        "binary": "二值变量 (Binary)",
        "ordinal": "有序分类 (Ordinal)",
        "id": "ID 列 (Identifier)",
        "text": "文本列 (Text)",
    }

    with st.expander(":material/tune: 变量类型覆盖", expanded=False):
        st.markdown(
            "如有变量的自动推断类型不准确，可在此手动更改。"
            "更改后点击「应用覆盖」按钮生效。"
        )

        # 初始化 session_state 中的覆盖记录
        if "type_overrides" not in st.session_state:
            st.session_state.type_overrides = {}

        # 为每个变量构建覆盖行
        override_data = []
        for v in variables:
            current_type = st.session_state.type_overrides.get(v.name, v.inferred_type)
            idx = type_options.index(current_type) if current_type in type_options else 0

            selected = st.selectbox(
                label=v.name,
                options=type_options,
                index=idx,
                format_func=lambda opt: type_labels.get(opt, opt),
                key=f"type_override_{v.name}",
                label_visibility="collapsed",
            )
            override_data.append((v.name, selected))

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                ":material/check: 应用覆盖",
                type="primary",
                use_container_width=True,
                key="apply_type_overrides",
            ):
                changed = 0
                for name, selected_type in override_data:
                    if selected_type != v.inferred_type:
                        st.session_state.type_overrides[name] = selected_type
                        # 更新 variables 列表中对应变量的 inferred_type
                        for var in st.session_state.variables:
                            if var.name == name:
                                var.inferred_type = selected_type
                                changed += 1

                if changed > 0:
                    st.success(f"已更新 {changed} 个变量的类型。")
                    st.rerun()
                else:
                    st.info("未检测到类型变更。")

        with col2:
            if st.button(
                ":material/clear: 重置所有覆盖",
                use_container_width=True,
                key="reset_type_overrides",
            ):
                if st.session_state.type_overrides:
                    st.session_state.type_overrides = {}
                    # 重新检测类型
                    from src.preprocessing.type_detector import VariableTypeDetector

                    df = st.session_state.get("data")
                    if df is not None:
                        detector = VariableTypeDetector()
                        st.session_state.variables = detector.detect(df)
                    st.success("类型覆盖已重置。")
                    st.rerun()

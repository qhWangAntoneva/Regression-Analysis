"""
Streamlit 可复用数据表格组件

提供数据预览和变量信息表的渲染功能。
已扩展：缺失值摘要、异常值检测。
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
                help=f"缺失值: {missing_count} / {n_rows} ({missing_count / max(n_rows, 1) * 100:.1f}%)",  # noqa: E501
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


def render_missing_value_summary(df: Any) -> dict[str, Any]:
    """渲染缺失值摘要表格。

    显示每个变量的缺失数量、缺失率及处理建议。使用颜色编码：
    - 绿色 (>=5%)：低缺失率
    - 黄色 (5-20%)：需要关注
    - 红色 (>20%)：严重缺失

    Args:
        df: pandas DataFrame。

    Returns:
        analyze() 返回的缺失值统计字典。
    """
    if st is None:
        return {}

    from src.preprocessing.missing import MissingValueHandler

    handler = MissingValueHandler()
    stats = handler.analyze(df)

    with st.expander(":material/table_chart: 缺失值摘要", expanded=True):
        col_data = stats["columns"]
        has_missing = any(v["count"] > 0 for v in col_data.values())

        if not has_missing:
            st.success("数据集中无缺失值。")
            return stats

        # 构建显示表格
        table_rows = []
        for col_name, info in col_data.items():
            if info["count"] == 0:
                continue
            pct = info["percentage"]

            # 颜色编码
            if pct > 20:
                level = "🔴 严重"
            elif pct > 5:
                level = "🟡 关注"
            else:
                level = "🟢 低"

            # 处理建议
            if pct > 20:
                suggestion = "建议删除该列或使用填充策略"
            elif pct > 5:
                suggestion = "建议使用均值/中位数填充"
            else:
                suggestion = "影响较小，可删除缺失行或填充"

            table_rows.append({
                "变量名": col_name,
                "数据类型": info["dtype"],
                "缺失数": info["count"],
                "缺失率": f"{pct:.2f}%",
                "等级": level,
                "处理建议": suggestion,
            })

        st.dataframe(table_rows, use_container_width=True)

        # 全局统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", stats["total_rows"])
        with col2:
            st.metric("总列数", stats["total_columns"])
        with col3:
            st.metric("总缺失值", stats["total_missing"])

    return stats


def render_outlier_detection_ui(df: Any) -> dict[str, Any] | None:
    """渲染异常值检测界面。

    提供"检测异常值"按钮，对选中的数值变量执行 IQR 异常值检测。
    显示每个变量的异常值数量，并可切换显示/隐藏异常值行。

    Args:
        df: pandas DataFrame。

    Returns:
        若执行检测，返回 flag_outliers 的 summary 字典；否则返回 None。
    """
    if st is None:
        return None

    from src.preprocessing.outliers import OutlierDetector

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        with st.expander(":material/outlier: 异常值检测", expanded=False):
            st.info("数据集中没有数值列可供检测。")
        return None

    with st.expander(":material/outlier: 异常值检测", expanded=False):
        st.markdown("选择要检测异常值的数值变量，点击按钮执行 IQR 异常值检测。")

        selected_cols = st.multiselect(
            "选择数值变量",
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            key="outlier_select_cols",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            multiplier = st.number_input(
                "IQR 倍数",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="1.5 为标准 IQR 方法，3.0 为极端值检测",
                key="outlier_multiplier",
            )

        with col2:
            method = st.radio(
                "检测方法",
                options=["iqr", "zscore"],
                index=0,
                horizontal=True,
                key="outlier_method",
                help="IQR：基于四分位距；Z-Score：基于标准差",
                format_func=lambda x: {"iqr": "IQR 方法", "zscore": "Z-Score 方法"}.get(x, x),
            )

        summary: dict[str, Any] | None = None

        if st.button(":material/search: 检测异常值", type="primary", use_container_width=True, key="detect_outliers"):  # noqa: E501
            if not selected_cols:
                st.warning("请至少选择一个数值变量。")
                return None

            with st.spinner("正在检测异常值..."):
                detector = OutlierDetector()
                try:
                    threshold_val = 3.0 if method == "zscore" else multiplier
                    kwargs = {"multiplier": threshold_val} if method == "iqr" else {"threshold": threshold_val}  # noqa: E501
                    _df_result, summary = detector.flag_outliers(
                        df, selected_cols, method=method, **kwargs
                    )
                except Exception as e:
                    st.error(f"异常值检测失败: {e}")
                    return None

        if summary:
            # 显示结果表格
            result_rows = []
            for col_name, info in summary.items():
                if "error" in info:
                    result_rows.append({
                        "变量名": col_name,
                        "异常值数": "N/A",
                        "异常率": "N/A",
                        "状态": f"⚠️ {info['error']}",
                    })
                else:
                    pct = info["percentage"]
                    if pct > 10:
                        level = "🔴 较多"
                    elif pct > 2:
                        level = "🟡 少量"
                    else:
                        level = "🟢 正常"

                    result_rows.append({
                        "变量名": col_name,
                        "异常值数": info["n_outliers"],
                        "异常率": f"{pct:.2f}%",
                        "状态": level,
                    })

            st.subheader("检测结果")
            st.dataframe(result_rows, use_container_width=True)

        return summary


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

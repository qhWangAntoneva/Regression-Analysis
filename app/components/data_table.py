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

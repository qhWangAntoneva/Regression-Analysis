# encoding: utf-8
"""
导出对话框组件

提供导出选项面板和结果展示功能。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


def render_export_options(key_prefix: str = "export") -> dict[str, Any] | None:
    """渲染导出选项面板。

    提供：
    - 导出格式选择（CSV / Excel / 图表PNG / 全部打包）
    - 导出路径设置
    - 导出内容勾选（系数表 / 描述统计 / 诊断图 / 模型摘要文本）
    - 导出按钮

    Args:
        key_prefix: Streamlit 组件 key 前缀。

    Returns:
        点击"导出"按钮后返回选项字典，否则返回 None。
        字典包含: format, path, include_coefficients, include_stats,
        include_charts, include_summary。
    """
    if st is None:
        return None

    st.subheader("导出设置")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox(
                "导出格式",
                options=["csv", "excel", "png", "all"],
                format_func=_format_label,
                key=f"{key_prefix}_format",
                help="选择导出格式：CSV（表格）、Excel（表格）、PNG（图表）或全部打包",
            )

        with col2:
            default_path = str(Path.cwd() / "exports")
            export_path = st.text_input(
                "导出路径",
                value=default_path,
                key=f"{key_prefix}_path",
                help="设置导出文件的保存目录",
            )

        # 导出内容勾选
        st.markdown("**导出内容**")
        content_cols = st.columns(4)

        with content_cols[0]:
            include_coefs = st.checkbox(
                "系数表",
                value=True,
                key=f"{key_prefix}_coefs",
                help="导出回归系数表",
            )

        with content_cols[1]:
            include_stats = st.checkbox(
                "描述统计",
                value=False,
                key=f"{key_prefix}_stats",
                help="导出描述性统计",
            )

        with content_cols[2]:
            include_charts = st.checkbox(
                "诊断图",
                value=True,
                key=f"{key_prefix}_charts",
                help="导出残差诊断图",
            )

        with content_cols[3]:
            include_summary = st.checkbox(
                "模型摘要",
                value=True,
                key=f"{key_prefix}_summary",
                help="导出模型摘要文本",
            )

        # 导出按钮
        clicked = st.button(
            ":material/download: 导出",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_button",
        )

    if clicked:
        return {
            "format": export_format,
            "path": export_path,
            "include_coefficients": include_coefs,
            "include_stats": include_stats,
            "include_charts": include_charts,
            "include_summary": include_summary,
        }

    return None


def render_export_result(filepaths: dict[str, str]) -> None:
    """渲染导出结果。

    显示成功导出的文件列表和文件大小信息。

    Args:
        filepaths: {文件类型: 文件路径} 字典。
    """
    if st is None:
        return

    if not filepaths:
        st.warning("导出结果为空。")
        return

    st.subheader("导出结果")

    # 分类：成功和失败
    success: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for file_type, filepath in filepaths.items():
        if filepath.startswith("导出失败"):
            failed.append((file_type, filepath))
        else:
            success.append((file_type, filepath))

    # 显示成功文件
    if success:
        with st.container(border=True):
            st.success(f"成功导出 {len(success)} 个文件")

            result_data = []
            for file_type, filepath in success:
                fpath = Path(filepath)
                size_bytes = fpath.stat().st_size if fpath.exists() else 0
                size_str = _format_file_size(size_bytes)
                result_data.append(
                    {
                        "文件类型": file_type,
                        "文件路径": str(fpath),
                        "文件大小": size_str,
                    }
                )

            st.dataframe(result_data, use_container_width=True, hide_index=True)

    # 显示失败信息
    if failed:
        with st.container(border=True):
            st.error(f"以下 {len(failed)} 个文件导出失败")
            for file_type, err_msg in failed:
                st.warning(f"{file_type}: {err_msg}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_label(fmt: str) -> str:
    """将格式键转换为中文标签。"""
    labels = {
        "csv": "CSV (.csv)",
        "excel": "Excel (.xlsx)",
        "png": "图表 PNG (.png)",
        "all": "全部打包",
    }
    return labels.get(fmt, fmt)


def _format_file_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

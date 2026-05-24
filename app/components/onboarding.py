# encoding: utf-8
"""
用户引导组件

提供首次使用引导、错误提示增强和内联帮助功能。
"""

from __future__ import annotations

from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


def render_first_run_guide() -> None:
    """首次使用 3 步引导弹窗。

    使用 st.popover 显示引导步骤。首次访问时自动展开，
    之后可通过 session_state 控制不再显示。
    """
    if st is None:
        return

    # 检查是否已完成引导
    if st.session_state.get("onboarding_done", False):
        return

    # 首次使用检查：如果 session_state 中有数据，说明不是首次使用
    has_data = st.session_state.get("data") is not None

    if has_data:
        # 已有数据，不显示首次引导
        st.session_state.onboarding_done = True
        return

    with st.popover(
        ":material/moving: 欢迎使用回归分析工具！点击查看快速入门",
        use_container_width=True,
    ):
        st.markdown("""
        ### 快速入门 — 3 步完成回归分析

        本工具帮助您快速完成回归建模、诊断和结果导出。

        ---

        **步骤 1：上传数据**

        在「数据上传」页面上传 CSV 或 Excel 文件。
        支持 UTF-8 和 GBK 编码的 CSV 文件。
        也可以点击「加载示例数据集」体验功能。

        **步骤 2：设定模型**

        在「模型设定」页面选择因变量和自变量。
        支持连续变量、分类变量、截距项和稳健标准误。

        **步骤 3：查看结果并导出**

        在「回归结果」页面查看系数表、诊断图和统计量。
        在「导出与报告」页面一键导出所有结果。

        ---

        **支持的模型类型**
        - 普通最小二乘法 (OLS) 线性回归
        - 支持分类变量（自动编码为虚拟变量）
        - 异方差稳健标准误 (HC3)
        """)

        # 标记已读
        if st.button("我知道了", type="primary", use_container_width=True):
            st.session_state.onboarding_done = True
            st.rerun()


def render_error_message(error: Exception) -> None:
    """中文错误消息渲染。

    提供用户友好的中文错误提示和修正建议，
    不暴露 Python traceback。

    Args:
        error: 捕获到的异常对象。
    """
    if st is None:
        return

    error_str = str(error)

    # 常见错误类型匹配
    if "ImportError" in type(error).__name__ or "ModuleNotFoundError" in type(error).__name__:
        st.error("依赖模块未安装。请运行以下命令安装依赖：")
        st.code("pip install -e .", language="bash")
        st.info(f"缺少的模块: {error_str}")

    elif "FileNotFoundError" in type(error).__name__:
        st.error("文件未找到。请检查文件路径是否正确。")
        st.info(f"路径: {error_str}")

    elif "KeyError" in type(error).__name__ or "列" in error_str:
        st.error("数据列不存在。请检查变量名是否正确。")
        st.info(f"详细信息: {error_str}")

    elif "ValueError" in type(error).__name__:
        st.error("数据或参数错误。")

        # 提供更具体的修正建议
        if "不足" in error_str:
            st.info("数据点太少。请提供更多观测数据，或减少自变量数量。")
        elif "缺失" in error_str:
            st.info("数据中存在缺失值。请检查数据并处理缺失值后重试。")
        elif "空" in error_str:
            st.info("输入数据为空。请检查数据是否正确加载。")
        elif "NaN" in error_str or "nan" in error_str.lower():
            st.info("数据包含非数值。请检查数据是否正确加载。")
        else:
            st.info(f"详细信息: {error_str}")

    elif "MemoryError" in type(error).__name__:
        st.error("内存不足。请尝试以下方案：")
        st.markdown("- 减少数据量（仅加载必要列）")
        st.markdown("- 关闭其他占用内存的程序")

    elif "ZeroDivisionError" in type(error).__name__ or "singular" in error_str.lower():
        st.error("模型存在完全多重共线性。请检查变量之间是否存在高度相关。")
        st.info("建议移除高度相关的自变量后重试。")

    else:
        # 通用错误处理（不暴露 traceback）
        st.error("操作执行时遇到错误。请检查输入数据是否正确。")
        st.info(f"错误信息: {error_str}")


def render_help_tooltip(text: str) -> None:
    """内联帮助提示。

    使用 st.markdown 显示一个小型帮助提示。

    Args:
        text: 帮助文本内容。
    """
    if st is None:
        return

    if text:
        st.markdown(
            f"""
            <div style="
                font-size: 0.85em;
                color: #666;
                background-color: #f5f5f5;
                padding: 4px 8px;
                border-radius: 4px;
                border-left: 3px solid #1f77b4;
                margin: 4px 0 8px 0;
            ">
            💡 {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

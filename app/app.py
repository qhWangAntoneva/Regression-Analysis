# encoding: utf-8
"""Regression Analysis -- Streamlit main entry point."""

from __future__ import annotations

import streamlit as st

from app.config import configure_page

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ---------------------------------------------------------------------------
configure_page()

# ---------------------------------------------------------------------------
# Crash Recovery: check for unclosed session
# ---------------------------------------------------------------------------
from src.utils.persistence import load_session, session_cache_exists, clear_session


def _check_crash_recovery() -> None:
    """Check for previously saved session state and offer recovery.

    Runs once per app startup.  If a .session_cache file exists,
    shows a banner with "恢复" and "忽略" buttons.
    """
    if not session_cache_exists():
        return

    # Only prompt once per session
    if st.session_state.get("_crash_recovery_handled"):
        return

    st.session_state._crash_recovery_handled = True

    # Use a sidebar info box
    with st.sidebar:
        st.info(":material/report: 检测到上次未正常关闭的会话。")

        col_rec, col_ign = st.columns(2)
        with col_rec:
            if st.button(":material/restore: 恢复", key="crash_restore", use_container_width=True):
                saved = load_session()
                if saved:
                    for key, value in saved.items():
                        if key != "_crash_recovery_handled":
                            st.session_state[key] = value
                    st.toast("会话已恢复！", icon="✅")
                    clear_session()
                else:
                    st.warning("无可恢复的会话数据。")

        with col_ign:
            if st.button(":material/delete: 忽略", key="crash_ignore", use_container_width=True):
                clear_session()
                st.rerun()


_check_crash_recovery()

# ---------------------------------------------------------------------------
# Auto-save session after model runs
# ---------------------------------------------------------------------------
if st.session_state.get("model_run_time"):
    from src.utils.persistence import save_session

    saveable = {
        "data_summary": st.session_state.get("data_summary"),
        "filename": st.session_state.get("filename"),
        "encoding": st.session_state.get("encoding"),
        "model_run_time": st.session_state.get("model_run_time"),
    }
    save_session(saveable)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem; margin: 0;">📊</h1>
        <h2 style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            Regression Analysis
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Define available pages using st.Page (Streamlit >=1.35)
pages = {
    "Data": [
        st.Page(
            "app/pages/01_data_upload.py",
            title="Upload & Preview",
            icon=":inbox_tray:",
            default=True,
        ),
    ],
    "Explore": [
        st.Page(
            "app/pages/02_data_explore.py",
            title="Data Explore",
            icon=":bar_chart:",
        ),
    ],
    "Analysis": [
        st.Page(
            "app/pages/03_model_spec.py",
            title="Model Specification",
            icon=":gear:",
        ),
        st.Page(
            "app/pages/04_model_results.py",
            title="Results",
            icon=":test_tube:",
        ),
    ],
    "Results": [
        st.Page(
            "app/pages/06_export.py",
            title="Export Report",
            icon=":page_facing_up:",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()

# ---------------------------------------------------------------------------
# Accessibility settings
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander(":material/accessibility: 辅助功能", expanded=False):
    cb_mode = st.toggle(
        "色盲友好模式",
        value=st.session_state.get("colorblind_mode", False),
        key="colorblind_mode",
        help=(
            "启用后使用蓝色/橙色配色替代默认的红/绿色。\n"
            "适用于红绿色盲（deuteranopia/protanopia）用户。"
        ),
    )
    st.caption(
        ":material/palette: 启用后，显著性高亮色从绿色改为蓝色系，"
        "相关系数热力图使用 Cividis 色阶。"
    )

    st.divider()
    st.markdown("#### :material/keyboard: 键盘快捷方式")
    st.caption(
        "**Tab** — 在表单元素间移动焦点\n"
        "**Enter/Space** — 激活按钮/复选框\n"
        "**Ctrl+F** — 浏览器页面搜索\n"
        "**上下箭头** — 滑动滑块和选择框\n"
        "按 **Tab** 可顺序到达：因变量 → 自变量 → 高级选项 → 运行按钮"
    )

# ---------------------------------------------------------------------------
# Sidebar project info
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Version:** 1.0.0

    **GitHub:** [qhWangAntoneva/Regression-Analysis](https://github.com/qhWangAntoneva/Regression-Analysis)
    """
)

# ---------------------------------------------------------------------------
# Global CSS for responsive layout and visual consistency
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Ensure dataframes and tables scroll horizontally on narrow screens */
    .stDataFrame > div[data-testid="stHorizontalBlock"] {
        overflow-x: auto !important;
    }
    /* Smooth scrolling for all scrollable containers */
    .stDataFrame {
        scroll-behavior: smooth;
    }
    /* Consistent card spacing between result cards and gallery cards */
    [data-testid="stExpander"] details {
        border-radius: 8px;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    /* Ensure bordered containers have consistent padding */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1rem !important;
    }
    /* Keyboard focus indicator for better accessibility */
    *:focus-visible {
        outline: 2px solid #1f77b4 !important;
        outline-offset: 2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# @st.cache_data decorator for expensive operations
# ---------------------------------------------------------------------------
import hashlib

from src.utils.persistence import clear_session, load_session, save_session, session_cache_exists


@st.cache_data(ttl=3600, show_spinner="正在计算相关系数矩阵...")
def cached_correlation_matrix(df_values: tuple, columns: tuple) -> tuple:
    """缓存相关系数矩阵计算结果。

    Args:
        df_values: DataFrame 的 numpy 数组展平的元组（用于缓存键）。
        columns: 列名元组。

    Returns:
        (列名列表, 相关系数矩阵嵌套列表) 的元组。
    """
    import numpy as np

    arr = np.array(df_values)
    corr = np.corrcoef(arr.T)
    return (list(columns), corr.tolist())

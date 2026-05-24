# encoding: utf-8
"""Streamlit page configuration."""

import streamlit as st


def configure_page() -> None:
    """Set the global Streamlit page configuration.

    Must be called once at the top of the main entry point.
    """
    st.set_page_config(
        page_title="Regression Analysis",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/qhWangAntoneva/Regression-Analysis",
            "Report a bug": "https://github.com/qhWangAntoneva/Regression-Analysis/issues",
            "About": "## Regression Analysis\n\nInteractive regression modeling and diagnostics.",
        },
    )

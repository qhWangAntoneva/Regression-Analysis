# encoding: utf-8
"""Streamlit page configuration and color scheme management."""

from __future__ import annotations

from typing import Dict

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


def get_color_scheme() -> Dict[str, str]:
    """Return the active color scheme based on accessibility settings.

    When ``st.session_state.colorblind_mode`` is ``True``, returns a
    color-blind-safe palette using blue/orange hues instead of the
    default red/green palette.  This ensures that significance
    highlighting in coefficient tables and dot-whisker plots remains
    distinguishable for users with deuteranopia/protanopia.

    Returns:
        A dictionary with keys:
          * ``sig_high_bg`` — background color for p < 0.01
          * ``sig_med_bg`` — background color for p < 0.05
          * ``sig_high`` — dot-whisker plot color for p < 0.01
          * ``sig_med`` — dot-whisker plot color for p < 0.05
          * ``sig_low`` — dot-whisker plot color for p < 0.1
          * ``no_sig`` — dot-whisker plot color for p >= 0.1
          * ``corr_colorscale`` — Plotly colorscale for correlation heatmap
          * ``plot_template`` — Plotly layout template name
    """
    if st.session_state.get("colorblind_mode", False):
        # Color-blind-safe palette — uses blue/orange instead of green/red.
        # Deuteranopia-safe: blue is distinguishable for all common CVD types.
        return {
            "sig_high_bg": "#bdd7e7",       # p<0.01 — medium blue background
            "sig_med_bg": "#deebf7",        # p<0.05 — light blue background
            "sig_high": "#2c7bb6",          # p<0.01 dot — teal-blue
            "sig_med": "#abd9e9",           # p<0.05 dot — pale blue
            "sig_low": "#fdae61",           # p<0.1  dot — orange
            "no_sig": "#bababa",            # non-sig dot — gray
            "corr_colorscale": "Cividis",   # perceptually uniform & CVD-safe
            "plot_template": "plotly_white",
        }
    # Default palette — green for significance (traditional regression output).
    return {
        "sig_high_bg": "#c8e6c9",          # p<0.01 — dark green
        "sig_med_bg": "#e8f5e9",           # p<0.05 — light green
        "sig_high": "darkgreen",           # p<0.01 dot — dark green
        "sig_med": "green",                # p<0.05 dot — green
        "sig_low": "orange",               # p<0.1  dot — orange
        "no_sig": "gray",                  # non-sig dot — gray
        "corr_colorscale": "RdBu_r",       # red-blue diverging
        "plot_template": "plotly_white",
    }

"""Visualization module.

Provides interactive plotly-based diagnostic and regression plots:
coefficient dot-whisker plots, ROC curves, odds-ratio forest plots,
residual diagnostics, and scatter plots with OLS regression lines.
"""

from src.visualization.coefficient import coefficient_plot, coefficient_plot_single
from src.visualization.logit_plots import odds_ratio_plot, roc_curve_plot
from src.visualization.residual import (
    cooks_distance_plot,
    diagnostic_dashboard,
    qq_plot,
    residual_vs_fitted_plot,
    scale_location_plot,
)
from src.visualization.scatter import scatter_with_regression

__all__ = [
    "coefficient_plot",
    "coefficient_plot_single",
    "roc_curve_plot",
    "odds_ratio_plot",
    "residual_vs_fitted_plot",
    "qq_plot",
    "scale_location_plot",
    "cooks_distance_plot",
    "diagnostic_dashboard",
    "scatter_with_regression",
]

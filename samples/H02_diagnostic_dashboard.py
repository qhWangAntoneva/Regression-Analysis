"""
H02: Diagnostic Dashboard for Housing OLS
==========================================

Learning objectives:
  - Generate all 4 diagnostic plots via diagnostic_dashboard().
  - Understand residual patterns, normality, heteroskedasticity, and
    influential observations.
  - Save each plot as an interactive HTML file.

The 4 plots:
  1. Residuals vs Fitted (check linearity & homoskedasticity)
  2. Q-Q Plot (check normality of residuals)
  3. Scale-Location Plot (check homoskedasticity)
  4. Cook's Distance (check influential observations)

Dataset: load_housing_data() — 500 obs OLS.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.utils.sample_data import load_housing_data
from src.visualization.residual import diagnostic_dashboard

# ---------------------------------------------------------------------------
# 1. Load data and fit OLS
# ---------------------------------------------------------------------------
data = load_housing_data()

spec = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
)

fitter = ModelFitter()
result = fitter.fit(spec, data, alpha=0.05)

print("=" * 60)
print("  H02: Diagnostic Dashboard (Housing OLS)")
print("=" * 60)
print(result.summary())
print()

# ---------------------------------------------------------------------------
# 2. Attach residuals & fitted values so the diagnostic functions can find them
# ---------------------------------------------------------------------------
# The visualization code (residual.py) looks for result.residuals and
# result.fitted_values — these are NOT automatically set by ModelResult,
# so we copy them from the raw statsmodels object.
result.residuals = result._raw_model.resid
result.fitted_values = result._raw_model.fittedvalues

# ---------------------------------------------------------------------------
# 3. Generate diagnostic dashboard (all 4 plots)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Generating Diagnostic Dashboard...")
print("=" * 60)

figs = diagnostic_dashboard(result, data)

plot_names = {
    "residual_fitted": "residuals_vs_fitted",
    "qq": "qq_plot",
    "scale_location": "scale_location",
    "cooks_distance": "cooks_distance",
}

for key, filename in plot_names.items():
    if key in figs:
        filepath = f"{filename}.html"
        figs[key].write_html(filepath)
        print(f"  Saved: {filepath}")
    else:
        print(f"  WARNING: {key} not available in dashboard output.")

print()

# ---------------------------------------------------------------------------
# 4. Interpret each diagnostic
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Diagnostic Interpretation Guide")
print("=" * 60)
print()
print("  1. Residuals vs Fitted:")
print("     - Red LOESS line should be roughly horizontal at y=0.")
print("     - A 'U' shape would suggest non-linearity in the model.")
print("     - A 'funnel' pattern would suggest heteroskedasticity.")
print()
print("  2. Q-Q Plot:")
print("     - Points should follow the red diagonal line closely.")
print("     - Systematic deviations indicate non-normality (e.g., heavy tails).")
print()
print("  3. Scale-Location Plot:")
print("     - LOESS line should be roughly horizontal.")
print("     - An upward/downward trend suggests heteroskedasticity.")
print()
print("  4. Cook's Distance:")
print("     - Red dashed line = 4/n threshold for influential observations.")
print("     - Bars above the line are potentially influential points.")
print("     - Investigate these observations for data errors or unusual cases.")
print()

print("Done. Open the .html files in a browser to view the interactive plots.")

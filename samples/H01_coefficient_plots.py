"""
H01: Coefficient Dot-Whisker Plots
====================================

Learning objectives:
  - Visualize a single model's coefficients with coefficient_plot_single().
  - Compare coefficients across nested models with coefficient_plot().
  - Understand how adding controls changes coefficient estimates.

Single model: housing OLS with sqft, bedrooms, age, location_score, floor, has_garage.
Multi-model: 3 nested wage models (progressively adding controls).
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
from src.utils.sample_data import load_housing_data, load_wages_data
from src.visualization.coefficient import coefficient_plot, coefficient_plot_single

# ===========================================================================
# Part 1: Single-model coefficient plot (housing data)
# ===========================================================================
print("=" * 60)
print("  Part 1: Single-Model Coefficient Plot (Housing OLS)")
print("=" * 60)

data_housing = load_housing_data()

spec_housing = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
)

fitter = ModelFitter()
result_housing = fitter.fit(spec_housing, data_housing, alpha=0.05)

print(result_housing.summary())
print()

fig_single = coefficient_plot_single(result_housing)
fig_single.write_html("coefficient_plot_single.html")
print("  Saved: coefficient_plot_single.html")
print()

# ===========================================================================
# Part 2: Multi-model comparison (3 nested wage models)
# ===========================================================================
print("=" * 60)
print("  Part 2: Multi-Model Coefficient Comparison (Wage Data)")
print("=" * 60)

data_wages = load_wages_data()

# Model 1: education only
spec1 = ModelSpec(
    dep_var="wage",
    indep_vars=["education"],
)

# Model 2: education + experience
spec2 = ModelSpec(
    dep_var="wage",
    indep_vars=["education", "experience"],
)

# Model 3: education + experience + hours + industry
spec3 = ModelSpec(
    dep_var="wage",
    indep_vars=["education", "experience"],
    control_vars=["hours_per_week", "industry"],
)

models = [spec1, spec2, spec3]
labels = ["Education Only", "+ Experience", "+ Hours + Industry"]

fitter2 = ModelFitter()
results_wage = fitter2.fit_multiple(models, data_wages, alpha=0.05)

for i, (label, res) in enumerate(zip(labels, results_wage)):
    print(f"\n  --- {label} ---")
    print(f"  R^2 = {res.r_squared:.4f}, Adj-R^2 = {res.adj_r_squared:.4f}")
    print(f"  AIC = {res.aic:.2f}, BIC = {res.bic:.2f}")

print()

# Generate multi-model comparison plot
fig_multi = coefficient_plot(results_wage, labels)
fig_multi.write_html("coefficient_plot_multi.html")
print("  Saved: coefficient_plot_multi.html")
print()

# ---------------------------------------------------------------------------
# Compare models as a DataFrame
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Model Comparison Table")
print("=" * 60)
from src.results.table import compare_models

comparison_df = compare_models(results_wage)
# Replace the superscript-2 character that causes GBK encoding issues
print(comparison_df.to_string().replace("\xb2", "^2"))
print()

print("Interpretation:")
print("  - As we add controls, the education coefficients may change")
print("    (omitted variable bias decreases with more controls).")
print("  - R^2 increases as we add predictors.")
print("  - The coefficient plot visually shows the stability/instability")
print("    of estimates across specifications (robustness check).")
print()

print("Done. Open the .html files in a browser to view the interactive plots.")

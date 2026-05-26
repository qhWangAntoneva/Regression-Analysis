"""
G01: VIF, Residual Tests, and Influence Diagnostics
=====================================================

Learning objectives:
  - Compute Variance Inflation Factor (VIF) for multicollinearity detection.
  - Run residual diagnostic tests (Shapiro-Wilk normality, Durbin-Watson autocorrelation).
  - Examine influence statistics (Cook's distance, leverage).

Dataset: load_housing_data() — 500 obs OLS model.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from src.modeling.diagnostics import influence_stats, residual_tests, vif
from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.utils.sample_data import load_housing_data

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
print("  G01: Diagnostic Tests on Housing OLS")
print("=" * 60)
print(result.summary())
print()

# ---------------------------------------------------------------------------
# 2. Variance Inflation Factor (VIF)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Variance Inflation Factor (VIF)")
print("=" * 60)
print("  Rule of thumb: VIF > 10 indicates serious multicollinearity;")
print("  VIF > 5 is a warning sign in conservative settings.")
print()

vif_df = vif(data, spec)
print(vif_df.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 3. Residual diagnostic tests
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Residual Diagnostic Tests")
print("=" * 60)

# Extract residuals from the raw statsmodels object
residuals = result._raw_model.resid
test_results = residual_tests(residuals)

print("  Shapiro-Wilk normality test:")
print(f"    Statistic: {test_results['shapiro_stat']:.4f}")
print(f"    p-value:   {test_results['shapiro_pvalue']:.4f}")
print(f"    Normal?    {test_results['shapiro_normal']}")
print("    (H0: residuals are normally distributed)")
print()
print("  Durbin-Watson autocorrelation test:")
print(f"    DW statistic: {test_results['dw_stat']:.4f}")
print(f"    Interpretation: {test_results['dw_autocorrelation']}")
print("    (DW = 2 means no autocorrelation; < 1 or > 3 is concerning)")
print()

# ---------------------------------------------------------------------------
# 4. Influence statistics (Cook's distance, leverage)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Influence Statistics (top 5 observations)")
print("=" * 60)

inf_df = influence_stats(result._raw_model)
print(f"  Total observations: {len(inf_df)}")
print()

# Top 5 by Cook's distance
top_cooks = inf_df.sort_values("cooks_d", ascending=False).head(5)
print("  Top 5 influential observations (by Cook's distance):")
print("  {:>5}  {:>12}  {:>10}".format("Obs", "Cook's D", "Leverage"))
print("  " + "-" * 35)
for _, row in top_cooks.iterrows():
    print(f"  {int(row['observation']):>5}  {row['cooks_d']:>12.6f}  {row['leverage']:>10.6f}")

print()
print("  Cook's distance > 4/n is often flagged as influential.")
threshold = 4.0 / len(inf_df)
n_influential = (inf_df["cooks_d"] > threshold).sum()
print(f"  4/n threshold: {threshold:.4f}")
print(f"  Observations exceeding threshold: {n_influential}")
print()

# High leverage observations (leverage > 2*p/n)
p = len(spec.all_predictors) + 1  # +1 for intercept
high_leverage_threshold = 2 * p / len(inf_df)
n_high_leverage = (inf_df["leverage"] > high_leverage_threshold).sum()
print(f"  High leverage threshold (2*p/n): {high_leverage_threshold:.4f}")
print(f"  Observations with high leverage: {n_high_leverage}")
print()

print("Done. G01 complete.")

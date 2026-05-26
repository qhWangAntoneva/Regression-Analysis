"""
A05: OLS Without Intercept
===========================

Learning objectives:
  - Fit an OLS model with has_intercept=False.
  - Understand the implications: R-squared is computed differently
    (uncentered), and interpretation changes (coefficients force
    prediction through origin).

Dataset: load_housing_data() — 500 obs.
Model: price ~ sqft + bedrooms + age + location_score + floor + has_garage - 1
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter
from src.utils.sample_data import load_housing_data

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_housing_data()
print("=" * 60)
print("  Dataset: load_housing_data()")
print(f"  Shape:   {data.shape[0]} rows x {data.shape[1]} columns")
print("  Model:   no intercept (forced through origin)")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 2. ModelSpec without intercept
# ---------------------------------------------------------------------------
spec_no_intercept = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
    has_intercept=False,
)

# Also fit with intercept for comparison
spec_with_intercept = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
    has_intercept=True,
)

fitter = ModelFitter()

# ---------------------------------------------------------------------------
# 3. Fit both models
# ---------------------------------------------------------------------------
result_ni = fitter.fit(spec_no_intercept, data, alpha=0.05)
result_w = fitter.fit(spec_with_intercept, data, alpha=0.05)

print("=" * 60)
print("  Model WITHOUT Intercept")
print("=" * 60)
print(result_ni.summary())
print()

print("=" * 60)
print("  Model WITH Intercept (for comparison)")
print("=" * 60)
print(result_w.summary())
print()

# ---------------------------------------------------------------------------
# 4. Coefficient comparison
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Coefficient Comparison")
print("=" * 60)
print(f"  {'Variable':<25} {'No Intercept':>15} {'With Intercept':>15}")
print("  " + "- * 55")

coef_ni = {c.name: c for c in result_ni.coefficients}
coef_w = {c.name: c for c in result_w.coefficients}

all_names = set(list(coef_ni.keys()) + list(coef_w.keys()))
for name in sorted(all_names):
    v_ni = coef_ni.get(name)
    v_w = coef_w.get(name)
    c_ni = f"{v_ni.coef:.4f}" if v_ni else "N/A"
    c_w = f"{v_w.coef:.4f}" if v_w else "N/A"
    print(f"  {name:<25} {c_ni:>15} {c_w:>15}")

print()
print(f"  R^2 (no intercept):     {result_ni.r_squared:.4f}")
print(f"  R^2 (with intercept):   {result_w.r_squared:.4f}")
print()
print("  Notes:")
print("  - Without an intercept, the regression is forced through (0,0).")
print("  - R-squared uses an 'uncentered' total SS in the no-intercept case,")
print("    so it is NOT directly comparable to the with-intercept R-squared.")
print("  - Coefficients can shift substantially when the intercept is removed,")
print("    especially if the true DGP has a non-zero intercept.")
print()

print("Done.")

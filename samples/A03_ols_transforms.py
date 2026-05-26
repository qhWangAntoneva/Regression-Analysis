"""
A03: OLS with Variable Transforms (Log, Square)
=================================================

Learning objectives:
  - Apply variable transformations via ModelSpec.transforms.
  - Understand log-level and squared-term interpretation.
  - See how transformed columns appear in the design matrix.

Dataset: load_housing_data() — 500 obs.
Transforms:
  - sqft -> log   (log-level: % change in price per unit change in sqft)
  - age  -> square (quadratic: diminishing marginal effect of age)
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
print("  Transforms: sqft -> log, age -> square")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 2. ModelSpec with transforms
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
    transforms={
        "sqft": "log",
        "age": "square",
    },
)

print("  --- Specification ---")
print(f"  dep_var:    {spec.dep_var}")
print(f"  predictors: {spec.all_predictors}")
print(f"  transforms: {spec.transforms}")
print(f"  -> sqft is replaced by sqft_log in the design matrix")
print(f"  -> age  is replaced by age_sq   in the design matrix")
print()

# ---------------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data, alpha=0.05)

print(result.summary())
print()

# ---------------------------------------------------------------------------
# 4. Coefficient table
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Coefficient Table (with transforms)")
print("=" * 60)
coef_df = result.to_dataframe()
print(coef_df.to_string())
print()

# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Interpretation (transformed variables)")
print("=" * 60)

# Find log_sqft coefficient
for c in result.coefficients:
    if "sqft_log" in c.name:
        # Log-level: 100 * (exp(coef) - 1) ≈ coef*100 for small coef
        pct = (np.exp(c.coef) - 1) * 100
        print(f"  log_sqft (beta = {c.coef:.4f}):")
        print(f"    -> A 1% increase in sqft is associated with a {c.coef/100:.4f} unit change in price.")
        print(f"    -> More precisely, a 1-unit increase in log_sqft multiplies price by exp({c.coef:.4f}) = {np.exp(c.coef):.4f}.")
        print(f"    -> p-value = {c.pvalue:.4f} {c.significance}")
        break

for c in result.coefficients:
    if "age_sq" in c.name:
        print(f"\n  age_sq (beta = {c.coef:.4f}):")
        print(f"    -> The linear age term captures the slope at age = 0.")
        print(f"    -> The squared term (age_sq) captures curvature.")
        print(f"    -> The marginal effect of age changes with age: d(price)/d(age) = beta_age + 2*beta_agesq*age")
        print(f"    -> p-value = {c.pvalue:.4f} {c.significance}")
        break

r2 = result.r_squared
if r2 is not None:
    print(f"\n  R-squared = {r2:.4f} — model fit with transformed variables.")
print()

print("Done.")

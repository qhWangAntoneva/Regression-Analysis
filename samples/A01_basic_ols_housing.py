"""
A01: Basic OLS on Housing Data
================================

Learning objectives:
  - Load a built-in dataset (housing).
  - Build a simple ModelSpec with dep_var, indep_vars.
  - Fit via ModelFitter and inspect result.summary() and result.to_dataframe().
  - Read and interpret the coefficient table.

Dataset: load_housing_data() — 500 obs, price ~ sqft + bedrooms + age + ...
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so that "from src import ..." works
# when the script is run directly as "uv run python samples/XX_name.py".
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.utils.sample_data import load_housing_data

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_housing_data()
print("=" * 60)
print("  Dataset: load_housing_data()")
print(f"  Shape:   {data.shape[0]} rows x {data.shape[1]} columns")
print(f"  Columns: {list(data.columns)}")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 2. Define model specification
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
)

print(f"  Formula: {spec.dep_var} ~ {' + '.join(spec.all_predictors)}")
print(f"  Model type: {spec.model_type}")
print()

# ---------------------------------------------------------------------------
# 3. Fit the model
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data, alpha=0.05)

# ---------------------------------------------------------------------------
# 4. Full summary
# ---------------------------------------------------------------------------
print(result.summary())
print()

# ---------------------------------------------------------------------------
# 5. Coefficient table as DataFrame
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Coefficient Table (DataFrame)")
print("=" * 60)
coef_df = result.to_dataframe()
print(coef_df.to_string())
print()

# ---------------------------------------------------------------------------
# 6. Plain-language interpretation
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Interpretation (OLS)")
print("=" * 60)

# Helper: find coefficient by name substring
def find_coef(result, keyword):
    for c in result.coefficients:
        if keyword.lower() in c.name.lower():
            return c
    return None

sqft_c = find_coef(result, "sqft")
bed_c = find_coef(result, "bedrooms")
age_c = find_coef(result, "age")
r2 = result.r_squared

if sqft_c:
    print(f"  - Each additional sq.ft. adds ${sqft_c.coef:.2f} to price (p={sqft_c.pvalue:.4f}).")
if bed_c:
    print(f"  - Each extra bedroom adds ${bed_c.coef:.2f} to price (p={bed_c.pvalue:.4f}).")
if age_c:
    print(f"  - Each year of age reduces price by ${abs(age_c.coef):.2f} (p={age_c.pvalue:.4f}).")
if r2 is not None:
    print(f"  - The model explains {r2:.2%} of price variance (R-squared).")
print()

print("Done. All outputs printed above.")

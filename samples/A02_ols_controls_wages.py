"""
A02: OLS with Control Variables on Wages Data
==============================================

Learning objectives:
  - Use categorical control variables (gender, industry).
  - Understand how control variables isolate the effect of primary predictors.
  - Interpret coefficients in the presence of controls.

Dataset: load_wages_data() — 400 obs.
Model: wage ~ education + experience + hours_per_week
  Controls: gender, industry
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
from src.utils.sample_data import load_wages_data

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_wages_data()
print("=" * 60)
print("  Dataset: load_wages_data()")
print(f"  Shape:   {data.shape[0]} rows x {data.shape[1]} columns")
print(f"  Columns: {list(data.columns)}")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 2. ModelSpec with primary predictors and control variables
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="wage",
    indep_vars=["education", "experience", "hours_per_week"],
    control_vars=["gender", "industry"],
)

print("  --- Model Specification ---")
print(f"  dep_var:     {spec.dep_var}")
print(f"  indep_vars:  {spec.indep_vars}")
print(f"  control_vars:{spec.control_vars}")
print(f"  All predictors: {spec.all_predictors}")
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
print("  Coefficient Table")
print("=" * 60)
coef_df = result.to_dataframe()
print(coef_df.to_string())
print()

# ---------------------------------------------------------------------------
# 5. Interpretation of key coefficients
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Interpretation (with controls)")
print("=" * 60)

for c in result.coefficients:
    name = c.name
    # Skip Intercept
    if name.lower() == "intercept":
        continue
    sig_str = c.significance if c.significance else "(not significant)"
    print(f"  {name:<35} coef={c.coef:>8.2f}  p={c.pvalue:.4f}  {sig_str}")

print()
print("  Notes:")
print("  - 'gender' and 'industry' are categorical; statsmodels auto-encodes")
print("    them as dummy variables with one reference category.")
print("  - The education dummies (relative to '高中以下') show the wage premium")
print("    of each education level, holding experience, hours, gender, and")
print("    industry constant.")
print()

print("Done.")

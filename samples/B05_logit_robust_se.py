"""
B05: Logit with Robust Standard Error Comparison
==================================================

Learning objectives:
  - Fit the same logit model with 5 different SE types.
  - Compare standard errors side-by-side in a table.
  - Understand how different robust estimators (HC0-HC3) differ.

Uses synthetic binary data similar to B01.
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

# ---------------------------------------------------------------------------
# 1. Synthetic binary data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=2028)
n = 500

x1 = rng.uniform(0, 10, n)
x2 = rng.normal(50, 15, n)
x3 = rng.choice([0, 1], n, p=[0.4, 0.6])

log_odds = -2.0 + 0.5 * x1 + 0.03 * x2 + 0.8 * x3
prob = 1.0 / (1.0 + np.exp(-log_odds))
y = rng.binomial(1, prob)

data = pd.DataFrame({
    "y": y,
    "x1": x1.round(2),
    "x2": x2.round(1),
    "x3": x3,
})

print("=" * 60)
print("  Synthetic Binary Data")
print("=" * 60)
print(f"  N = {len(data)}")
print(f"  y=1: {y.sum()} ({y.mean()*100:.1f}%)")
print("  Predictors: x1 (continuous), x2 (continuous), x3 (binary)")
print()

# ---------------------------------------------------------------------------
# 2. ModelSpec
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="y",
    indep_vars=["x1", "x2", "x3"],
    model_type="logit",
)

# ---------------------------------------------------------------------------
# 3. Fit with all 5 SE types
# ---------------------------------------------------------------------------
cov_types = ["nonrobust", "HC0", "HC1", "HC2", "HC3"]
fitter = ModelFitter()

results: dict[str, object] = {}
for ct in cov_types:
    result = fitter.fit(spec, data, alpha=0.05, cov_type=ct)
    results[ct] = result

# ---------------------------------------------------------------------------
# 4. Side-by-side SE comparison
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Standard Error Comparison (Logit)")
print("=" * 60)

# Collect all variable names (minus Intercept)
var_names = [c.name for c in results["nonrobust"].coefficients if c.name.lower() != "intercept"]

header = f"  {'Variable':<20}"
for ct in cov_types:
    header += f" {ct:>12}"
print(header)
print("  " + "-" * (20 + 13 * len(cov_types)))

for vname in var_names:
    row = f"  {vname:<20}"
    se_vals = []
    for ct in cov_types:
        res = results[ct]
        for c in res.coefficients:
            if c.name == vname:
                row += f" {c.se:>12.6f}"
                se_vals.append(c.se)
                break
    print(row)

print()
print("  HC0-HC3 are heteroskedasticity-consistent estimators.")
print("  HC1 includes a finite-sample correction (n/(n-k)).")
print("  HC2 and HC3 are more conservative (larger SEs when")
print("  there are high-leverage observations).")
print()

# ---------------------------------------------------------------------------
# 5. Coefficient table from the HC1 model
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Full Model Results (HC1 robust SE)")
print("=" * 60)
print(results["HC1"].summary())
print()

# ---------------------------------------------------------------------------
# 6. Odds ratios for HC1
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Odds Ratios (HC1 model)")
print("=" * 60)
res_hc1 = results["HC1"]
for c in res_hc1.coefficients:
    if c.name.lower() == "intercept":
        continue
    or_val = np.exp(c.coef)
    se = c.se
    print(f"  {c.name:8} {c.coef:7.4f} {se:7.4f} {or_val:7.4f} {c.pvalue:7.4f} {c.significance}")
print()

print("Done.")

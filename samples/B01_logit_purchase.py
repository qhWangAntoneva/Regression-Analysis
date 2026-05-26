"""
B01: Logit Model on Synthetic Binary Data
==========================================

Learning objectives:
  - Generate synthetic binary outcome data for a logit model.
  - Fit a logit model and interpret log-odds coefficients.
  - Compute and interpret odds ratios (exp(coef)).
  - Understand pseudo-R-squared and likelihood-ratio test.

True DGP: logit(P) = -3 + 0.05 * income + 0.1 * age
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

# ---------------------------------------------------------------------------
# 1. Generate synthetic binary data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=2026)
n = 500

income = rng.uniform(20, 150, n)       # income in thousands
age = rng.uniform(18, 80, n)           # age in years

# True logit DGP: logit(P) = -3 + 0.05*income + 0.1*age
log_odds = -3.0 + 0.05 * income + 0.1 * age
prob = 1.0 / (1.0 + np.exp(-log_odds))
purchase = rng.binomial(1, prob)

data = pd.DataFrame({
    "purchase": purchase,
    "income": income.round(1),
    "age": age.round(1),
})

print("=" * 60)
print("  Synthetic Binary Data (Logit DGP)")
print("=" * 60)
print(f"  N = {len(data)}")
print(f"  purchase = 1: {purchase.sum()} ({purchase.mean()*100:.1f}%)")
print(f"  purchase = 0: {(1-purchase).sum()} ({(1-purchase.mean())*100:.1f}%)")
print(f"  income range: [{income.min():.1f}, {income.max():.1f}]")
print(f"  age range:    [{age.min():.1f}, {age.max():.1f}]")
print()

# ---------------------------------------------------------------------------
# 2. ModelSpec
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="purchase",
    indep_vars=["income", "age"],
    model_type="logit",
)

print(f"  Formula: logit({spec.dep_var}) ~ {' + '.join(spec.all_predictors)}")
print()

# ---------------------------------------------------------------------------
# 3. Fit logit model
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data, alpha=0.05)

print(result.summary())
print()

# ---------------------------------------------------------------------------
# 4. Coefficient table with odds ratios
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Coefficient Table (with Odds Ratios)")
print("=" * 60)
coef_df = result.to_dataframe()
print(coef_df.to_string())
print()

# ---------------------------------------------------------------------------
# 5. Odds ratio interpretation
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Interpretation (Logit - Odds Ratios)")
print("=" * 60)

for c in result.coefficients:
    if c.name.lower() == "intercept":
        continue
    or_val = np.exp(c.coef)
    print(f"  {c.name}:")
    print(f"    Log-OR = {c.coef:.4f}, OR = {or_val:.4f}")
    if c.name == "income":
        print(f"    -> Each additional unit of income multiplies the odds of purchase by {or_val:.4f}.")
    elif c.name == "age":
        print(f"    -> Each additional year of age multiplies the odds of purchase by {or_val:.4f}.")
    print(f"    -> 95% CI for OR: [{np.exp(c.ci_lower):.4f}, {np.exp(c.ci_upper):.4f}]")
    print(f"    -> p-value: {c.pvalue:.4f} {c.significance}")
    print()

# Check pseudo R²
if result.pseudo_r_squared is not None:
    print(f"  McFadden's pseudo R^2 = {result.pseudo_r_squared:.4f}")
    print(f"  LR chi2 = {result.llr:.4f} (p = {result.llr_pvalue:.6e})")
print()

print("Done.")

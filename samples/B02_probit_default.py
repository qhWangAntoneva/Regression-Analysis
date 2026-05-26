"""
B02: Probit Default Prediction Model
======================================
Fit a probit model on synthetic binary data (default prediction).
DGP: probit(P) = -2 + 0.8 * credit_score_std + 0.5 * debt_ratio
Interpretation focuses on z-statistics and pseudo R-squared.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'from src...' imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec

print("=" * 60)
print("  B02: Probit Default Prediction Model")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(202)
n = 500

credit_score_std = rng.normal(0, 1, n)
debt_ratio = rng.uniform(0.05, 0.8, n)

from scipy.stats import norm

# Linear index on the probit scale
# probit(P) = Phi^{-1}(P), so P = Phi(linear_index)
z = -2.0 + 0.8 * credit_score_std + 0.5 * debt_ratio
p = norm.cdf(z)
default = rng.binomial(1, p)

data = pd.DataFrame({
    "default": default,
    "credit_score_std": credit_score_std,
    "debt_ratio": debt_ratio,
})

print(f"\nGenerated {len(data)} observations.")
print(f"Default rate: {default.mean():.2%}")
print()

# ---------------------------------------------------------------------------
# 2. Define model specification
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="default",
    indep_vars=["credit_score_std", "debt_ratio"],
    model_type="probit",
)

# ---------------------------------------------------------------------------
# 3. Fit model
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data, alpha=0.05)

print("\n" + "=" * 60)
print("  Probit Model Summary")
print("=" * 60)
print(result.summary())

# ---------------------------------------------------------------------------
# 4. Extract key statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Key Statistics & Interpretation")
print("=" * 60)

pr2 = f"{result.pseudo_r_squared:.4f}" if result.pseudo_r_squared is not None else "N/A"
print(f"\n  Pseudo R-squared:  {pr2}")
print(f"  Log-Likelihood:    {result.log_likelihood:.4f}")
print(f"  AIC:               {result.aic:.4f}")
print(f"  BIC:               {result.bic:.4f}")
print(f"  Observations:      {result.n_obs}")
print()

df_coef = result.to_dataframe()
for var_name in df_coef.index:
    coef = df_coef.loc[var_name, "系数"]
    se = df_coef.loc[var_name, "标准误"]
    zval = df_coef.loc[var_name, "z值"]
    pv = df_coef.loc[var_name, "p值"]
    stars = df_coef.loc[var_name, "显著性"]
    print(f"  {var_name:<12s} {coef:>9.4f}  {se:>7.4f}  {zval:>7.4f}  {pv:>7.4f}  {stars}")

# ---------------------------------------------------------------------------
# 5. Plain-language interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - credit_score_std has a positive and significant effect on default")
print("    probability. A 1 SD increase raises the probit index by ~0.8.")
print("  - debt_ratio is also positive: higher debt burden increases default risk.")
print("  - The pseudo R-squared indicates the model's explanatory power.")
print("-" * 60)
print("\nDone. (B02)")

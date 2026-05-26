"""
B04: Logit vs Probit Comparison
=================================
Fit both logit and probit models on the same synthetic binary outcome data.
Compare AIC, BIC, and coefficient patterns using compare_models().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter
from src.results.table import compare_models

print("=" * 60)
print("  B04: Logit vs Probit Comparison")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Generate synthetic binary outcome data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(204)
n = 800

income_std = rng.normal(0, 1, n)
age_std = rng.normal(0, 1, n)

# Linear index
z = -1.5 + 0.7 * income_std + 0.4 * age_std
p = norm.cdf(z)
y = rng.binomial(1, p)

data = pd.DataFrame({
    "y": y,
    "income_std": income_std,
    "age_std": age_std,
})

print(f"\nGenerated {n} observations, positive rate = {y.mean():.2%}")
print()

# ---------------------------------------------------------------------------
# 2. Define specs
# ---------------------------------------------------------------------------
spec_logit = ModelSpec(
    dep_var="y",
    indep_vars=["income_std", "age_std"],
    model_type="logit",
)

spec_probit = ModelSpec(
    dep_var="y",
    indep_vars=["income_std", "age_std"],
    model_type="probit",
)

# ---------------------------------------------------------------------------
# 3. Fit both models
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result_logit = fitter.fit(spec_logit, data)
result_probit = fitter.fit(spec_probit, data)

print("Logit summary:")
print(result_logit.summary())

print("\nProbit summary:")
print(result_probit.summary())

# ---------------------------------------------------------------------------
# 4. Compare
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Model Comparison Table")
print("=" * 60)

comparison_df = compare_models([result_logit, result_probit])
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Detailed comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Detailed AIC / BIC Comparison")
print("=" * 60)
print(f"  Logit  - AIC: {result_logit.aic:.2f}  BIC: {result_logit.bic:.2f}")
print(f"  Probit - AIC: {result_probit.aic:.2f}  BIC: {result_probit.bic:.2f}")

delta_aic = result_logit.aic - result_probit.aic
delta_bic = result_logit.bic - result_probit.bic
print(f"\n  Delta (Logit - Probit):  AIC = {delta_aic:+.2f}  BIC = {delta_bic:+.2f}")

if abs(delta_aic) < 2:
    print("  The models fit similarly — AIC difference < 2.")
elif delta_aic < 0:
    print("  Logit has lower AIC (preferred).")
else:
    print("  Probit has lower AIC (preferred).")

# ---------------------------------------------------------------------------
# 6. Interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - Logit and probit typically produce similar fit statistics")
print("    when the true DGP is close to normal (probit link).")
print("  - Coefficients differ in scale: logit ~= 1.6 * probit coefficients.")
print("  - Both correctly identify income_std and age_std as significant.")
print("-" * 60)
print("\nDone. (B04)")

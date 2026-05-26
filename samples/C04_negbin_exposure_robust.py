"""
C04: Negative Binomial with Exposure + HC1 Robust Standard Errors

Demonstrates a Negative Binomial rate model with exposure variable and
HC1 (heteroskedasticity-robust) standard errors.

DGP: log(frequency / hours) = -1.5 + 0.6 * skill_score + 0.3 * workload

We fit with cov_type="HC1" to produce standard errors that are robust
to misspecification of the variance function.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src import ...` works
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec

# ── Generate synthetic data (errors / defects per logged hour) ───────────
rng = np.random.default_rng(20241104)
N = 250

skill_score = rng.uniform(0, 10, N)   # operator skill (0=novice, 10=expert)
workload = rng.uniform(1, 5, N)       # task complexity score
log_hours = rng.uniform(3, 8, N)      # log of exposure hours

# True DGP: Negative Binomial with overdispersion
alpha_nb = 0.6
log_rate = -1.5 + 0.6 * skill_score + 0.3 * workload
log_mu = log_rate + log_hours
nu = rng.gamma(1 / alpha_nb, alpha_nb, N)
defects = rng.poisson(np.exp(log_mu) * nu)

df = pd.DataFrame({
    "defects": defects,
    "skill_score": skill_score.round(2),
    "workload": workload.round(2),
    "log_hours": log_hours.round(3),
})

print("=" * 60)
print("  C04: NegBin + Exposure + HC1 Robust SE")
print("=" * 60)
print(f"\n  Data shape: {df.shape}")
print(f"  Defects: min={defects.min()}, max={defects.max()}, "
      f"mean={defects.mean():.2f}")
print(f"  Variance/Mean: {defects.var() / defects.mean():.2f}")

# ── Build spec ───────────────────────────────────────────────────────────
spec = ModelSpec(
    dep_var="defects",
    indep_vars=["skill_score", "workload"],
    model_type="negbin",
    exposure_var="log_hours",
)

fitter = ModelFitter()
result = fitter.fit(spec, df, alpha=0.05, cov_type="HC1")

print("\n" + "=" * 60)
print("  Model Summary")
print("=" * 60)
print(f"\n  Model type:        {result.model_type}")
print(f"  Observations:       {result.n_obs}")
print("  Exposure var:       log_hours (offset)")
print(f"  Std. Error type:    {getattr(result, 'se_type', 'HC1')}")
print(f"  Log-Likelihood:     {result.log_likelihood:.4f}")
print(f"  AIC:                {result.aic:.4f}")
print(f"  BIC:                {result.bic:.4f}")
disp = getattr(result, "dispersion", None)
if disp is not None:
    print(f"  Dispersion (α):     {disp:.4f}")

# ── Coefficient table ────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Coefficient Table (HC1 robust SE)")
print("-" * 60)
tbl = result.to_dataframe()
print(tbl.to_string())

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  HC1 robust standard errors guard against variance-function")
print("  misspecification in the count model.")
for c in result.coefficients:
    if c.name != "Intercept":
        irr = np.exp(c.coef)
        print(f"  {c.name}: coef = {c.coef:.4f}, SE(robust) = {c.se:.4f}, "
              f"IRR = {irr:.4f}, p = {c.pvalue:.4f}{c.significance}")

if disp is not None:
    print(f"\n  The dispersion parameter α = {disp:.4f} > 0 confirms")
    print("  overdispersion, justifying NegBin over Poisson in this rate model.")
else:
    print("\n  Dispersion parameter not available from this engine path.")
    print("  The Gamma-Poisson mixture DGP ensures overdispersion is present.")

print("\n" + "=" * 60)
print("  Done — C04 complete.")
print("=" * 60)

"""
C03: Poisson Regression with Exposure (Rate Model)

Demonstrates Poisson regression with an exposure / offset variable.
The response is accident counts; the exposure is log_miles driven.
The model estimates the rate: log(accidents / miles) = Xb, equivalently
log(accidents) = offset(log_miles) + Xb.

DGP: log(rate) = -2 + 0.4 * driving_exp_std + 0.8 * risk_factor
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

# ── Generate synthetic accident data ─────────────────────────────────────
rng = np.random.default_rng(20241103)
N = 300

driving_exp = rng.uniform(1, 40, N)
driving_exp_std = (driving_exp - driving_exp.mean()) / driving_exp.std()
risk_factor = rng.uniform(0, 1, N)  # 0=low risk, 1=high risk
log_miles = rng.uniform(6, 10, N)   # log of miles driven
miles = np.exp(log_miles)

# Rate per mile: log(rate) = -2 + 0.4*exp_std + 0.8*risk
log_rate = -2.0 + 0.4 * driving_exp_std + 0.8 * risk_factor
log_lambda = log_rate + log_miles  # offset
accidents = rng.poisson(np.exp(log_lambda))

df = pd.DataFrame({
    "accidents": accidents,
    "driving_exp_std": driving_exp_std.round(3),
    "risk_factor": risk_factor.round(3),
    "log_miles": log_miles.round(3),
    "miles": miles.round(0).astype(int),
})

print("=" * 60)
print("  C03: Poisson with Exposure — Accident Rate Model")
print("=" * 60)
print(f"\n  Data shape: {df.shape}")
print(f"  Accidents: min={accidents.min()}, max={accidents.max()}, "
      f"mean={accidents.mean():.2f}")
print(f"  Miles (mean): {miles.mean():.0f}")
print(f"  Crude rate (accidents / 1000 miles): "
      f"{(accidents.sum() / miles.sum() * 1000):.4f}")

# ── Build spec with exposure_var ─────────────────────────────────────────
spec = ModelSpec(
    dep_var="accidents",
    indep_vars=["driving_exp_std", "risk_factor"],
    model_type="poisson",
    exposure_var="log_miles",
)

fitter = ModelFitter()
result = fitter.fit(spec, df, alpha=0.05)

print("\n" + "=" * 60)
print("  Model Summary")
print("=" * 60)
print(f"\n  Model type:      {result.model_type}")
print(f"  Observations:     {result.n_obs}")
print("  Exposure var:     log_miles (offset)")
print(f"  Log-Likelihood:   {result.log_likelihood:.4f}")
print(f"  AIC:              {result.aic:.4f}")
print(f"  BIC:              {result.bic:.4f}")

# ── Coefficient table ────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Coefficient Table (IRR = exp(coef) for rate interpretation)")
print("-" * 60)
tbl = result.to_dataframe()
print(tbl.to_string())

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  Coefficients are on the log-rate scale.  exp(coef) gives the")
print("  multiplicative effect on the accident rate per unit mile.")
for c in result.coefficients:
    if c.name != "Intercept":
        irr = np.exp(c.coef)
        pct = (irr - 1) * 100
        print(f"  {c.name}: a one-unit increase multiplies the accident "
              f"rate by {irr:.4f} ({pct:+.2f}%), p = "
              f"{c.pvalue:.4f}{c.significance}")

print("\n  The offset log_miles accounts for different exposure levels,")
print("  so coefficients reflect rate effects, not pure count effects.")

print("\n" + "=" * 60)
print("  Done — C03 complete.")
print("=" * 60)

"""
C02: Negative Binomial Regression — Insurance Claims

Demonstrates Negative Binomial regression on overdispersed count data
(insurance claim counts).  The NegativeBinomial family includes a
dispersion parameter that captures extra-Poisson variability.

We show the estimated dispersion parameter and model fit statistics (AIC).
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src import ...` works
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter

# ── Generate overdispersed count data ────────────────────────────────────
rng = np.random.default_rng(20241102)
N = 400

# Simulate from a Negative Binomial via Gamma-Poisson mixture:
#   lambda_i = exp(Xb) * nu_i,  nu_i ~ Gamma(1/alpha, alpha)
age = rng.normal(45, 12, N)
car_value = rng.uniform(5, 50, N)  # in 10k USD
prior_claims = rng.poisson(0.3, N)
young_driver = rng.choice([0, 1], N, p=[0.8, 0.2])

log_mu = (
    -0.5
    + 0.02 * age
    + 0.04 * car_value
    + 0.3 * prior_claims
    + 0.8 * young_driver
)
alpha = 0.8  # dispersion parameter — larger = more overdispersion
nu = rng.gamma(1 / alpha, alpha, N)
claims = rng.poisson(np.exp(log_mu) * nu)

df = pd.DataFrame({
    "claims": claims,
    "age": age.round(1),
    "car_value": car_value.round(2),
    "prior_claims": prior_claims,
    "young_driver": young_driver,
})

print("=" * 60)
print("  C02: Negative Binomial — Insurance Claims")
print("=" * 60)
print(f"\n  Data shape: {df.shape}")
print(f"  Claims: min={claims.min()}, max={claims.max()}, "
      f"mean={claims.mean():.2f}, var={claims.var():.2f}")
print(f"  Variance/Mean ratio: {claims.var() / claims.mean():.2f}  "
      f"(>1 suggests overdispersion)")

# ── Build spec & fit ─────────────────────────────────────────────────────
spec = ModelSpec(
    dep_var="claims",
    indep_vars=["age", "car_value", "prior_claims", "young_driver"],
    model_type="negbin",
)

fitter = ModelFitter()
result = fitter.fit(spec, df, alpha=0.05)

print("\n" + "=" * 60)
print("  Model Summary")
print("=" * 60)
print(f"\n  Model type:      {result.model_type}")
print(f"  Observations:     {result.n_obs}")
print(f"  Log-Likelihood:   {result.log_likelihood:.4f}")
print(f"  AIC:              {result.aic:.4f}")
print(f"  BIC:              {result.bic:.4f}")

# The dispersion field is set on the ModelResult by the count engine
dispersion = getattr(result, "dispersion", None)
if dispersion is not None:
    print(f"  Dispersion (α):   {dispersion:.4f}   "
          f"(>0 confirms overdispersion)")
else:
    print("  Dispersion:       N/A (check fitted model)")

# ── Coefficient table ────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Coefficient Table (with IRR)")
print("-" * 60)
tbl = result.to_dataframe()
print(tbl.to_string())

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  Negative Binomial relaxes the Poisson assumption that")
print("  variance = mean.  The dispersion parameter α > 0 confirms")
print("  overdispersion, justifying NB over Poisson.")
for c in result.coefficients:
    if c.name != "Intercept":
        irr = np.exp(c.coef)
        pct = (irr - 1) * 100
        print(f"  {c.name}: IRR = {irr:.4f}  ({pct:+.2f}%), "
              f"p = {c.pvalue:.4f}{c.significance}")

print("\n" + "=" * 60)
print("  Done — C02 complete.")
print("=" * 60)

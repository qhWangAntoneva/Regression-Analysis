"""
C01: Poisson Regression — Hospital Visits

Demonstrates Poisson regression on synthetic count data (number of hospital
visits in the past year).  The DGP is a log-linear count model:
    log(visits) = 0.5 + 0.3 * age_std + 0.5 * health_score - 0.2 * smoker

IRR (Incidence Rate Ratio) = exp(coef) is reported for interpretability.
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

# ── Generate synthetic data ──────────────────────────────────────────────
rng = np.random.default_rng(20241101)
N = 300

age = rng.normal(50, 15, N)
age_std = (age - age.mean()) / age.std()
health_score = rng.uniform(0, 10, N)  # 0=poor, 10=excellent
smoker = rng.choice([0, 1], N, p=[0.65, 0.35])

log_lambda = (
    0.5
    + 0.3 * age_std
    + 0.5 * health_score
    - 0.2 * smoker
)
visits = rng.poisson(np.exp(log_lambda))

df = pd.DataFrame({
    "visits": visits,
    "age_std": age_std.round(3),
    "health_score": health_score.round(2),
    "smoker": smoker,
})

print("=" * 60)
print("  C01: Poisson Regression — Hospital Visits")
print("=" * 60)
print(f"\n  Data shape: {df.shape}")
print(f"  Visits: min={visits.min()}, max={visits.max()}, mean={visits.mean():.2f}")
print(f"  Smokers: {smoker.sum()}/{N} ({(smoker.sum()/N)*100:.0f}%)")

# ── Build spec & fit ─────────────────────────────────────────────────────
spec = ModelSpec(
    dep_var="visits",
    indep_vars=["age_std", "health_score", "smoker"],
    model_type="poisson",
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

# ── Coefficient table with IRR ───────────────────────────────────────────
print("\n" + "-" * 60)
print("  Coefficient Table (with IRR)")
print("-" * 60)
tbl = result.to_dataframe()
print(tbl.to_string())

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
for c in result.coefficients:
    irr = np.exp(c.coef)
    if c.name == "Intercept":
        print(f"  Baseline visit rate (intercept): IRR = {irr:.4f}")
    else:
        pct = (irr - 1) * 100
        print(f"  {c.name}: a one-unit increase is associated with a "
              f"{pct:+.2f}% change in expected visits (IRR = {irr:.4f}, "
              f"p = {c.pvalue:.4f}{c.significance})")

print("\n" + "=" * 60)
print("  Done — C01 complete.")
print("=" * 60)

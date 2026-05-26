"""
E01: Panel Data — Fixed Effects (FE) on GDP Growth

Estimates a panel fixed-effects model on synthetic data for 30 provinces
over 10 years.  Entity fixed effects absorb time-invariant provincial
heterogeneity.

DGP: GDP = 5 + 0.3 * investment + 0.5 * education + 0.2 * infrastructure
          + province_FE + N(0,1)

where province_FE are province-specific intercept shifts drawn from N(0, 2).
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

# ── Generate panel data ──────────────────────────────────────────────────
rng = np.random.default_rng(20241111)

N_PROVINCES = 30
N_YEARS = 10

provinces = [f"P{p:02d}" for p in range(1, N_PROVINCES + 1)]
years = list(range(2010, 2010 + N_YEARS))

# Province-specific fixed effects
province_fe = {p: rng.normal(0, 2.0) for p in provinces}

rows = []
for prov in provinces:
    for yr in years:
        investment = rng.uniform(1, 10)
        education = rng.uniform(2, 8)
        infrastructure = rng.uniform(1, 6)
        gdp = (
            5.0
            + 0.3 * investment
            + 0.5 * education
            + 0.2 * infrastructure
            + province_fe[prov]
            + rng.normal(0, 1)
        )
        rows.append({
            "province": prov,
            "year": yr,
            "gdp": round(gdp, 2),
            "investment": round(investment, 2),
            "education": round(education, 2),
            "infrastructure": round(infrastructure, 2),
        })

df = pd.DataFrame(rows)
print("=" * 60)
print("  E01: Panel FE — Provincial GDP")
print("=" * 60)
print(f"\n  Panel dimensions: {N_PROVINCES} provinces x {N_YEARS} years")
print(f"  Total obs:        {len(df)}")
print(f"  GDP range:        {df['gdp'].min():.2f} – {df['gdp'].max():.2f}")

# ── Build panel FE spec ──────────────────────────────────────────────────
spec = ModelSpec(
    dep_var="gdp",
    indep_vars=["investment", "education", "infrastructure"],
    model_type="panel",
    entity_var="province",
    time_var="year",
    panel_model="fixed",
)

fitter = ModelFitter()
result = fitter.fit(spec, df, alpha=0.05)

print("\n" + "=" * 60)
print("  Model Summary (Panel FE)")
print("=" * 60)
print(f"\n  Panel type:       {result.method} ({getattr(result, 'panel_type', 'N/A')})")
print(f"  Observations:      {result.n_obs}")
print(f"  Entities:          {getattr(result, 'entity_count', 'N/A')}")
print(f"  Time periods:      {getattr(result, 'time_count', 'N/A')}")

within_r2 = getattr(result, "within_r_squared", None)
between_r2 = getattr(result, "between_r_squared", None)
overall_r2 = getattr(result, "overall_r_squared", None)
if within_r2 is not None:
    print(f"  Within R-squared:  {within_r2:.4f}")
if between_r2 is not None:
    print(f"  Between R-squared: {between_r2:.4f}")
if overall_r2 is not None:
    print(f"  Overall R-squared: {overall_r2:.4f}")
print(f"  Log-Likelihood:    {result.log_likelihood:.4f}" if result.log_likelihood is not None else "")
print(f"  AIC:               {result.aic:.4f}")
print(f"  BIC:               {result.bic:.4f}")

# ── Coefficient table ────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Coefficient Table")
print("-" * 60)
tbl = result.to_dataframe()
print(tbl.to_string())

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  FE controls for all time-invariant province characteristics.")
print("  Coefficients are identified from within-province variation over time.")
for c in result.coefficients:
    if c.name != "Intercept":
        print(f"  {c.name}: a one-unit within-province increase is associated "
              f"with a {c.coef:.4f} change in GDP (p = {c.pvalue:.4f}{c.significance})")

print("\n" + "=" * 60)
print("  Done — E01 complete.")
print("=" * 60)

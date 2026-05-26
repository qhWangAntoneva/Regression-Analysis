"""
E02: Panel Data — Random Effects (RE) on Firm Profitability

Estimates a panel random-effects model on synthetic data for 50 firms
over 8 years.  RE assumes that entity-specific effects are uncorrelated
with the regressors, allowing both within- and between-entity variation
to contribute to the estimates.

DGP: profit = 3 + 0.4 * R&D + 0.3 * market_share + 0.1 * size
            + firm_RE + year_RE + N(0,1)
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
rng = np.random.default_rng(20241112)

N_FIRMS = 50
N_YEARS = 8

firms = [f"F{f:03d}" for f in range(1, N_FIRMS + 1)]
years = list(range(2015, 2015 + N_YEARS))

# Firm and year random effects
firm_re = {f: rng.normal(0, 1.5) for f in firms}
year_re = {yr: rng.normal(0, 0.5) for yr in years}

rows = []
for firm in firms:
    rd = rng.uniform(0, 5)
    size = rng.uniform(10, 100)
    for yr in years:
        mkt_share = rng.uniform(0, 0.3)
        profit = (
            3.0
            + 0.4 * rd
            + 0.3 * mkt_share
            + 0.1 * size
            + firm_re[firm]
            + year_re[yr]
            + rng.normal(0, 1)
        )
        rows.append({
            "firm": firm,
            "year": yr,
            "profit": round(profit, 2),
            "rd": round(rd, 2),
            "market_share": round(mkt_share, 3),
            "size": round(size, 1),
        })

df = pd.DataFrame(rows)

print("=" * 60)
print("  E02: Panel RE — Firm Profitability")
print("=" * 60)
print(f"\n  Panel dimensions: {N_FIRMS} firms x {N_YEARS} years")
print(f"  Total obs:        {len(df)}")
print(f"  Profit range:     {df['profit'].min():.2f} – {df['profit'].max():.2f}")

# ── Build panel RE spec ──────────────────────────────────────────────────
spec = ModelSpec(
    dep_var="profit",
    indep_vars=["rd", "market_share", "size"],
    model_type="panel",
    entity_var="firm",
    time_var="year",
    panel_model="random",
)

fitter = ModelFitter()
result = fitter.fit(spec, df, alpha=0.05)

print("\n" + "=" * 60)
print("  Model Summary (Panel RE)")
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
print("  RE uses both within-firm and between-firm variation, making it")
print("  more efficient than FE if the RE assumption (no correlation")
print("  between effects and regressors) holds.")
for c in result.coefficients:
    if c.name != "Intercept":
        print(f"  {c.name}: a one-unit increase is associated with "
              f"a {c.coef:.4f} change in profit (p = "
              f"{c.pvalue:.4f}{c.significance})")

print("\n" + "=" * 60)
print("  Done — E02 complete.")
print("=" * 60)

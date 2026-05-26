"""
C05: Poisson vs Negative Binomial Comparison

Fits both Poisson and NegBin on overdispersed count data, then compares
the two models using AIC, BIC, and the NegBin dispersion parameter.

The `compare_models()` function produces a side-by-side table showing
coefficients and fit statistics for both models.
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
from src.results.table import compare_models

# ── Generate overdispersed data ──────────────────────────────────────────
rng = np.random.default_rng(20241105)
N = 350

x1 = rng.normal(0, 1, N)
x2 = rng.uniform(-1, 1, N)
cat = rng.choice(["A", "B", "C"], N, p=[0.4, 0.35, 0.25])

# True DGP: Negative Binomial (overdispersed)
true_alpha = 1.2
log_mu = 0.8 + 0.5*x1 - 0.3*x2 + np.where(cat == "B", 0.4, 0.0) + np.where(cat == "C", -0.2, 0.0)
nu = rng.gamma(1 / true_alpha, true_alpha, N)
y = rng.poisson(np.exp(log_mu) * nu)

df = pd.DataFrame({
    "y": y,
    "x1": x1.round(3),
    "x2": x2.round(3),
    "cat": cat,
})

print("=" * 60)
print("  C05: Poisson vs NegBin Comparison")
print("=" * 60)
print(f"\n  Data shape: {df.shape}")
print(f"  y: mean={y.mean():.2f}, var={y.var():.2f}, "
      f"V/M ratio={y.var() / y.mean():.2f}")
print(f"  True dispersion (α): {true_alpha}")

# ── Fit both models ──────────────────────────────────────────────────────
spec_poisson = ModelSpec(
    dep_var="y",
    indep_vars=["x1", "x2", "cat"],
    model_type="poisson",
)
spec_negbin = ModelSpec(
    dep_var="y",
    indep_vars=["x1", "x2", "cat"],
    model_type="negbin",
)

fitter = ModelFitter()
result_poi = fitter.fit(spec_poisson, df, alpha=0.05)
result_nb = fitter.fit(spec_negbin, df, alpha=0.05)

# ── Individual summaries ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Poisson Model")
print("=" * 60)
print(f"  AIC:   {result_poi.aic:.2f}")
print(f"  BIC:   {result_poi.bic:.2f}")
print(f"  Log-L: {result_poi.log_likelihood:.2f}")
print(result_poi.to_dataframe().to_string())

print("\n" + "=" * 60)
print("  Negative Binomial Model")
print("=" * 60)
print(f"  AIC:   {result_nb.aic:.2f}")
print(f"  BIC:   {result_nb.bic:.2f}")
print(f"  Log-L: {result_nb.log_likelihood:.2f}")
disp_nb = getattr(result_nb, "dispersion", None)
if disp_nb is not None:
    print(f"  Disp:  {disp_nb:.4f}")
print(result_nb.to_dataframe().to_string())

# ── Compare models ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Side-by-Side Comparison")
print("=" * 60)
comparison = compare_models([result_poi, result_nb])
print(comparison.to_string())

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
aic_delta = result_poi.aic - result_nb.aic
bic_delta = result_poi.bic - result_nb.bic
print(f"  Delta AIC (Poisson - NegBin): {aic_delta:.2f}")
print(f"  Delta BIC (Poisson - NegBin): {bic_delta:.2f}")
if aic_delta > 0:
    print("  NegBin has lower AIC and is preferred.")
else:
    print("  Poisson has lower AIC and is preferred.")
if disp_nb is not None and disp_nb > 0.1:
    print(f"  Dispersion α = {disp_nb:.4f} > 0 — overdispersion detected.")
    print("  NegBin is the appropriate model for this data.")
else:
    print("  Dispersion α ≈ 0 — Poisson may be adequate.")

print("\n" + "=" * 60)
print("  Done — C05 complete.")
print("=" * 60)

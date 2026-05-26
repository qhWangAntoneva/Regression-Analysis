"""
G02: Model Comparison with Nested Models (Wages Data)
======================================================
Fit three nested OLS models on wages data using fit_multiple().
Base:     wage ~ experience
Middle:   wage ~ experience + education
Full:     wage ~ experience + education + industry + gender
Compare AIC/BIC progression with compare_models().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.utils.sample_data import load_wages_data
from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter
from src.results.table import compare_models

print("=" * 60)
print("  G02: Nested Model Comparison — Wages")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_wages_data()
print(f"\nLoaded wages data: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Columns: {list(data.columns)}")
print()

# ---------------------------------------------------------------------------
# 2. Define three nested specifications
# ---------------------------------------------------------------------------
spec_base = ModelSpec(
    dep_var="wage",
    indep_vars=["experience"],
    missing_strategy="drop",
)

spec_edu = ModelSpec(
    dep_var="wage",
    indep_vars=["experience", "education"],
    missing_strategy="drop",
)

spec_full = ModelSpec(
    dep_var="wage",
    indep_vars=["experience", "education"],
    control_vars=["industry", "gender"],
    missing_strategy="drop",
)

specs = [spec_base, spec_edu, spec_full]

# ---------------------------------------------------------------------------
# 3. Fit all three models
# ---------------------------------------------------------------------------
fitter = ModelFitter()
results = fitter.fit_multiple(specs, data)

model_labels = ["Base (experience only)",
                "+ Education",
                "+ Industry + Gender"]

for label, result in zip(model_labels, results):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(result.summary())

# ---------------------------------------------------------------------------
# 4. Comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Horizontal Comparison Table")
print("=" * 60)

comparison_df = compare_models(results)
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. AIC / BIC progression
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  AIC / BIC Progression (lower is better)")
print("=" * 60)
print(f"  {'Model':<35s} {'AIC':>12s} {'BIC':>12s} {'N':>8s}")
print(f"  {'-'*67}")
for i, (label, result) in enumerate(zip(model_labels, results)):
    print(f"  {label:<35s} {result.aic:>12.2f} {result.bic:>12.2f} {result.n_obs:>8d}")

print(f"\n  AIC reduction from base to full: {results[0].aic - results[2].aic:.2f}")
print(f"  BIC reduction from base to full: {results[0].bic - results[2].bic:.2f}")

# ---------------------------------------------------------------------------
# 6. Interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - Adding education substantially improves model fit (large drop in AIC/BIC).")
print("  - Adding industry and gender provides further (smaller) improvement.")
print("  - The full model captures wage variation from human capital (education,")
print("    experience) and labor market segmentation (industry, gender).")
print("-" * 60)
print("\nDone. (G02)")

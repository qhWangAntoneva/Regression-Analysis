"""
A04: OLS with Interaction Terms and HC1 Robust Standard Errors
===============================================================
Use the built-in housing dataset. Model: price ~ sqft + bedrooms + sqft:has_garage.
Robust SE via cov_type='HC1'. Interpret the interaction coefficient.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.utils.sample_data import load_housing_data

print("=" * 60)
print("  A04: OLS — Interactions + HC1 Robust SE")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_housing_data()
print(f"\nLoaded housing data: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Columns: {list(data.columns)}")
print()

# ---------------------------------------------------------------------------
# 2. Specification with interaction
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="price",
    indep_vars=["sqft", "bedrooms", "location_score", "floor"],
    control_vars=["age", "has_garage"],
    interaction_terms=[("sqft", "has_garage")],
    missing_strategy="drop",
)

# ---------------------------------------------------------------------------
# 3. Fit with HC1 robust SE
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data, cov_type="HC1")

print(f"SE type: {result.se_type}")
print(result.summary())

# ---------------------------------------------------------------------------
# 4. Interpret the interaction term
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Interaction Coefficient Interpretation")
print("=" * 60)

df_coef = result.to_dataframe()
interaction_rows = [name for name in df_coef.index if "sqft" in name and "has_garage" in name]

if interaction_rows:
    iv = interaction_rows[0]
    coef_val = df_coef.loc[iv, "系数"]
    pval = df_coef.loc[iv, "p值"]
    stars = df_coef.loc[iv, "显著性"]

    print(f"\n  Interaction term:          {iv}")
    print(f"  Coefficient:               {coef_val:.4f}")
    print(f"  p-value:                   {pval:.4f} {stars}")

    if pval < 0.05:
        print("\n  The interaction is statistically significant (p < 0.05).")
        print("  The marginal effect of sqft on price differs between houses")
        print("  with and without a garage.")
    else:
        print("\n  The interaction is not statistically significant at 5%.")
else:
    print("\n  Interaction term not found in coefficient table.")

# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - HC1 robust SEs guard against heteroskedasticity.")
print("  - The sqft:has_garage interaction captures whether the sqft-price")
print("    relationship is moderated by garage presence.")
print("  - All coefficients, including the interaction, are reported with")
print("    heteroskedasticity-consistent standard errors.")
print("-" * 60)
print("\nDone. (A04)")

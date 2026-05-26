"""
E03: Hausman Test — FE vs RE Model Selection

The Hausman specification test compares Fixed Effects and Random Effects
estimates for panel data.  Under H0 (RE is consistent), both estimators
are consistent but RE is more efficient.  A significant p-value indicates
RE is inconsistent, so FE is preferred.

We use `run_hausman_from_results()` from the hausman module, which
extracts coefficient vectors and covariance matrices from the raw
linearmodels objects stored on each ModelResult.
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
from src.modeling.hausman import run_hausman_from_results
from src.modeling.specification import ModelSpec

# ── Generate panel data with entity FE in the DGP ────────────────────────
# Because the DGP includes entity fixed effects correlated with regressors,
# RE should be inconsistent and the Hausman test should favour FE.
# Design choices:
#   - 150 entities x 3 periods (short T so FE is less efficient than RE,
#     making Var(b_fe) - Var(b_re) more likely PSD).
#   - Entity FE has substantial variance and is correlated with x1.
#   - True x1 coefficient = 0.8, entity FE loaded through both x1 and y.
rng = np.random.default_rng(20241113)

N_ENTITIES = 150
N_PERIODS = 3

entities = [f"E{e:03d}" for e in range(1, N_ENTITIES + 1)]
years = list(range(2010, 2010 + N_PERIODS))

entity_fe = {e: rng.normal(3, 2) for e in entities}

rows = []
for ent in entities:
    for yr in years:
        # Correlate x1 with the entity FE: x1 depends on the FE
        x1 = 0.8 * entity_fe[ent] + rng.normal(0, 1)
        x2 = rng.uniform(0, 5)
        y = (
            1.0
            + 0.8 * x1
            - 0.4 * x2
            + entity_fe[ent]
            + rng.normal(0, 1.0)
        )
        rows.append({
            "entity": ent,
            "year": yr,
            "y": round(y, 2),
            "x1": round(x1, 2),
            "x2": round(x2, 2),
        })

df = pd.DataFrame(rows)

print("=" * 60)
print("  E03: Hausman Test — FE vs RE")
print("=" * 60)
print(f"\n  Panel dimensions: {N_ENTITIES} entities x {N_PERIODS} periods")
print(f"  Total obs:        {len(df)}")
print("  DGP includes entity FE correlated with x1 — RE should be invalid.")

# ── Fit FE model ─────────────────────────────────────────────────────────
spec_fe = ModelSpec(
    dep_var="y",
    indep_vars=["x1", "x2"],
    model_type="panel",
    entity_var="entity",
    time_var="year",
    panel_model="fixed",
)
spec_re = ModelSpec(
    dep_var="y",
    indep_vars=["x1", "x2"],
    model_type="panel",
    entity_var="entity",
    time_var="year",
    panel_model="random",
)

fitter = ModelFitter()
result_fe = fitter.fit(spec_fe, df, alpha=0.05)
result_re = fitter.fit(spec_re, df, alpha=0.05)

print("\n" + "=" * 60)
print("  FE Model Coefficients")
print("=" * 60)
print(result_fe.to_dataframe().to_string())

print("\n" + "=" * 60)
print("  RE Model Coefficients")
print("=" * 60)
print(result_re.to_dataframe().to_string())

# ── Hausman test ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Hausman Specification Test")
print("=" * 60)

h_result = run_hausman_from_results(result_fe, result_re)

if h_result is None:
    print("  Hausman test could not be computed (raw models unavailable).")
else:
    print(f"\n  Chi-squared statistic:  {h_result['statistic']:.4f}")
    print(f"  Degrees of freedom:     {h_result['df']}")
    print(f"  p-value:                {h_result['p_value']:.6f}")
    print(f"  Recommendation:         {h_result['recommendation']}")
    print(f"  Common variables:       {h_result['common_vars']}")

    if h_result["p_value"] < 0.05:
        print("\n  --> p < 0.05: RE is inconsistent. Use FE.")
    else:
        print("\n  --> p >= 0.05: RE is consistent and more efficient. Use RE.")

# ── Plain-language interpretation ────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  The Hausman test compares FE and RE coefficient vectors.")
print("  A significant p-value suggests that the random effects")
print("  assumption (E[effect | X] = 0) is violated, so FE is preferred.")
print("  This aligns with our DGP, which includes entity FE correlated")
print("  with the regressor x1.")

print("\n" + "=" * 60)
print("  Done — E03 complete.")
print("=" * 60)

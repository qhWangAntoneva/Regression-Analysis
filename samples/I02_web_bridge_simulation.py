"""
I02: Web Bridge Simulation

Demonstrates the data flow between the web frontend and the Python
backend.  In production, the bridge at `web/py/bridge.py` receives
JSON strings and returns JSON.  Here we simulate that flow:

  1. A JSON-like dict spec is constructed (as the frontend would send).
  2. The dict is converted to a ModelSpec.
  3. ModelFitter fits the model.
  4. The result is serialised back to a plain dict (as the bridge would
     return via json.dumps).

This shows how the bridge's `run_regression()` maps to the core API.
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

# ── Simulate the JSON spec a frontend would POST ─────────────────────────
# This is exactly what `web/py/bridge.py::run_regression()` receives as
# `spec_json`.
bridge_spec_dict = {
    "dep_var": "price",
    "indep_vars": ["sqft", "bedrooms", "location_score"],
    "has_intercept": True,
    "alpha": 0.05,
    "cov_type": "HC0",
    "missing_strategy": "drop",
    "model_type": "ols",
}

# ── Generate data (same way app.py would load from uploaded CSV) ─────────
rng = np.random.default_rng(20241122)
N = 200

sqft = rng.normal(1500, 500, N).clip(500, 5000)
bedrooms = rng.choice([2, 3, 4, 5], N, p=[0.2, 0.4, 0.3, 0.1])
location_score = rng.uniform(1, 10, N)

price = (
    50000
    + 200 * sqft
    + 15000 * bedrooms
    + 30000 * location_score
    + rng.normal(0, 40000, N)
).clip(50000, None)

df = pd.DataFrame({
    "price": price.round(0).astype(int),
    "sqft": sqft.round(0).astype(int),
    "bedrooms": bedrooms.astype(int),
    "location_score": location_score.round(2),
})

print("=" * 60)
print("  I02: Web Bridge Simulation")
print("=" * 60)
print("\n  Step 1: Frontend sends JSON spec")
print("-" * 40)
for k, v in bridge_spec_dict.items():
    print(f"    {k:20s} = {v}")

# ── Convert dict to ModelSpec (bridge internal step) ─────────────────────
print("\n  Step 2: Bridge converts dict -> ModelSpec")
print("-" * 40)
spec = ModelSpec(
    dep_var=bridge_spec_dict["dep_var"],
    indep_vars=bridge_spec_dict["indep_vars"],
    has_intercept=bridge_spec_dict["has_intercept"],
    missing_strategy=bridge_spec_dict["missing_strategy"],
    model_type=bridge_spec_dict["model_type"],
)
print(f"    ModelSpec created: {spec.dep_var} ~ {' + '.join(spec.indep_vars)}")

# ── Fit model (bridge internal step) ─────────────────────────────────────
print("\n  Step 3: Bridge calls ModelFitter.fit()")
print("-" * 40)
fitter = ModelFitter()
result = fitter.fit(
    spec,
    df,
    alpha=bridge_spec_dict["alpha"],
    cov_type=bridge_spec_dict["cov_type"],
)
print(f"    Observations:  {result.n_obs}")
print(f"    Parameters:    {result.n_params}")
print(f"    R-squared:     {result.r_squared:.4f}")
print(f"    AIC:           {result.aic:.4f}")

# ── Serialise result to plain dict (bridge return step) ──────────────────
print("\n  Step 4: Bridge serialises result -> dict (JSON-ready)")
print("-" * 40)
output = {
    "success": True,
    "model_type": result.model_type,
    "n_obs": result.n_obs,
    "r_squared": result.r_squared,
    "adj_r_squared": result.adj_r_squared,
    "aic": result.aic,
    "bic": result.bic,
    "se_type": getattr(result, "se_type", "nonrobust"),
    "coefficients": [
        {
            "name": c.name,
            "coef": round(c.coef, 6),
            "se": round(c.se, 6),
            "pvalue": c.pvalue,
            "ci_lower": round(c.ci_lower, 6),
            "ci_upper": round(c.ci_upper, 6),
            "significance": c.significance,
        }
        for c in result.coefficients
    ],
}

print("  Output dict keys:")
for k in output:
    if k != "coefficients":
        print(f"    {k:20s}: {output[k]}")
    else:
        print(f"    coefficients: [{len(output[k])} rows]")

print("\n  Coefficient detail:")
for coeff in output["coefficients"]:
    print(f"    {coeff['name']:20s} coef={coeff['coef']:>10.6f}  "
          f"se={coeff['se']:.6f}  p={coeff['pvalue']:.4f}  {coeff['significance']}")

# ── Interpretation ───────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  The bridge pattern works as follows in the deployed web app:")
print("    1. Frontend sends { data_json, spec_json } via POST.")
print("    2. web/py/bridge.py::run_regression() parses JSON, builds")
print("       the DataFrame and ModelSpec, and calls ModelFitter.fit().")
print("    3. The result is serialised to JSON and returned to the UI.")
print("  This sample simulates steps 2-4 without actual HTTP or Pyodide.")

print("\n" + "=" * 60)
print("  Done — I02 complete.")
print("=" * 60)

"""
D02: MixedLM on Longitudinal Data (Random Intercept)
====================================================
Fit a mixed-effects model on grouped time-series data.
DGP: outcome = 10 + 2*time + group_random_intercept + 0.5*group_random_slope*time + e
Group intercept ~ N(0,4), slope ~ N(0,1), e ~ N(0,5).

Note: The current framework estimates a random intercept (not random slope)
via the group_var parameter. The data is generated with both random intercepts
and slopes for realism, but only the intercept variance is estimated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter

print("=" * 60)
print("  D02: MixedLM — Random Intercept + Random Slope")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Generate data: 50 groups x 15 timepoints
# ---------------------------------------------------------------------------
rng = np.random.default_rng(220)

n_groups = 50
n_timepoints = 15
n_total = n_groups * n_timepoints

group_ids = np.repeat(np.arange(n_groups), n_timepoints)
time = np.tile(np.arange(n_timepoints), n_groups).astype(float)

# Group-level random effects
group_intercepts = rng.normal(0, np.sqrt(4), n_groups)
group_slopes = rng.normal(0, np.sqrt(1), n_groups)

gi_expanded = np.repeat(group_intercepts, n_timepoints)
gs_expanded = np.repeat(group_slopes, n_timepoints)

# Outcome
error = rng.normal(0, np.sqrt(5), n_total)
outcome = 10 + 2 * time + gi_expanded + gs_expanded * time + error

data = pd.DataFrame({
    "group_id": group_ids.astype(int),
    "time": time,
    "outcome": outcome.round(2),
})

print(f"\nData: {n_total} observations across {n_groups} groups, {n_timepoints} timepoints each")
print(f"Outcome mean: {data['outcome'].mean():.2f}")
print()

# ---------------------------------------------------------------------------
# 2. Specify mixed model with random slope
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="outcome",
    indep_vars=["time"],
    model_type="mixedlm",
    group_var="group_id",
)

# ---------------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data)

print(result.summary())

# ---------------------------------------------------------------------------
# 4. Random-effects output
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MixedLM — Random Effects")
print("=" * 60)
print(f"  Groups: {result.group_count}")
if result.re_var:
    print("  Variance components:")
    for k, v in result.re_var.items():
        print(f"    {k}: {v:.4f}")
print()

# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------
print("-" * 60)
print("  Interpretation:")
print("  - The fixed effect of time (~2.0) captures the average linear trend.")
print("  - The group random intercept captures between-group baseline differences.")
print("  - Data was generated with random slopes, but the current framework")
print("    estimates only random intercepts via the group_var parameter.")
print("  - The random intercept variance is still informative about group-level")
print("    heterogeneity after controlling for time.")
print("-" * 60)
print("\nDone. (D02)")

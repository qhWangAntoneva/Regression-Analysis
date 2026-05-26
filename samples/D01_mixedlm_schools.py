"""
D01: MixedLM with Random Intercept (School Data)
=================================================
Fit a mixed-effects model with random intercepts per school.
DGP: score = 50 + 5*study_hours + 3*parental_edu + school_intercept + e
School intercepts ~ N(0, 8), e ~ N(0, 10).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec

print("=" * 60)
print("  D01: MixedLM — Random Intercept (Schools)")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Generate hierarchical data: 100 schools x 20 students
# ---------------------------------------------------------------------------
rng = np.random.default_rng(210)

n_schools = 100
n_per_school = 20
n_total = n_schools * n_per_school

school_ids = np.repeat(np.arange(n_schools), n_per_school)

study_hours = rng.uniform(0, 15, n_total)
parental_edu = rng.choice([1, 2, 3, 4, 5], n_total)

# School-level random intercepts
school_re = rng.normal(0, np.sqrt(8), n_schools)
school_re_expanded = np.repeat(school_re, n_per_school)

# Outcome
error = rng.normal(0, np.sqrt(10), n_total)
score = 50 + 5 * study_hours + 3 * parental_edu + school_re_expanded + error

data = pd.DataFrame({
    "school_id": school_ids.astype(int),
    "study_hours": study_hours.round(2),
    "parental_edu": parental_edu.astype(int),
    "score": score.round(1),
})

print(f"\nData: {n_total} students in {n_schools} schools")
print(f"Score mean: {data['score'].mean():.1f}, std: {data['score'].std():.1f}")
print()

# ---------------------------------------------------------------------------
# 2. Specify mixed model
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="score",
    indep_vars=["study_hours", "parental_edu"],
    model_type="mixedlm",
    group_var="school_id",
)

# ---------------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result = fitter.fit(spec, data)

print(result.summary())

# ---------------------------------------------------------------------------
# 4. MixedLM-specific output
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MixedLM — Random Effects")
print("=" * 60)
print(f"  Group variable:          {result.group_var}")
print(f"  Number of groups:        {result.group_count}")
if result.re_var:
    print("  Random effects variance components:")
    for k, v in result.re_var.items():
        print(f"    {k}: {v:.4f}")
print()

# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------
print("-" * 60)
print("  Interpretation:")
print("  - Each additional hour of study is associated with ~5 point")
print("    increase in score (fixed effect).")
print("  - parental_edu shows ~3 point boost per education level.")
print("  - The school random intercept variance captures between-school")
print("    heterogeneity after controlling for observed predictors.")
print("-" * 60)
print("\nDone. (D01)")

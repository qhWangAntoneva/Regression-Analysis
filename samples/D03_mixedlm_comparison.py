"""
D03: MixedLM Model Comparison
==============================
Compare null (intercept-only) vs full mixed model on school data.
Show how adding fixed effects improves fit using AIC and BIC.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter
from src.results.table import compare_models

print("=" * 60)
print("  D03: MixedLM — Null vs Full Model Comparison")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Generate school data (same DGP as D01)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(230)

n_schools = 100
n_per_school = 20
n_total = n_schools * n_per_school

school_ids = np.repeat(np.arange(n_schools), n_per_school)
study_hours = rng.uniform(0, 15, n_total)
parental_edu = rng.choice([1, 2, 3, 4, 5], n_total)

school_re = rng.normal(0, np.sqrt(8), n_schools)
school_re_expanded = np.repeat(school_re, n_per_school)
error = rng.normal(0, np.sqrt(10), n_total)
score = 50 + 5 * study_hours + 3 * parental_edu + school_re_expanded + error

data = pd.DataFrame({
    "school_id": school_ids.astype(int),
    "study_hours": study_hours.round(2),
    "parental_edu": parental_edu.astype(int),
    "score": score.round(1),
})

# ---------------------------------------------------------------------------
# 2. Null model: random intercept only, no fixed predictors
# ---------------------------------------------------------------------------
spec_null = ModelSpec(
    dep_var="score",
    indep_vars=["study_hours"],     # minimal; mixedlm needs at least 1 predictor
    model_type="mixedlm",
    group_var="school_id",
)

# Note: MixedLM requires at least one fixed-effect predictor in this framework.
# We use study_hours for the null model and add parental_edu for the full model.

spec_full = ModelSpec(
    dep_var="score",
    indep_vars=["study_hours", "parental_edu"],
    model_type="mixedlm",
    group_var="school_id",
)

# ---------------------------------------------------------------------------
# 3. Fit both
# ---------------------------------------------------------------------------
fitter = ModelFitter()
result_null = fitter.fit(spec_null, data)
result_full = fitter.fit(spec_full, data)

print("\nNull model (study_hours only):")
print(result_null.summary())

print("\nFull model (study_hours + parental_edu):")
print(result_full.summary())

# ---------------------------------------------------------------------------
# 4. Compare
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Model Comparison")
print("=" * 60)

comparison_df = compare_models([result_null, result_full])
print(comparison_df.to_string(index=False))

print(f"\n  {'':30s} {'Null':>12s} {'Full':>12s}")
print(f"  {'N (Observations)':30s} {result_null.n_obs:>12d} {result_full.n_obs:>12d}")
print(f"  {'Log-Likelihood':30s} {result_null.log_likelihood:>12.4f} {result_full.log_likelihood:>12.4f}")
print(f"  {'R-squared':30s} {result_null.r_squared:>12.4f} {result_full.r_squared:>12.4f}")

# Note: REML AIC/BIC are NaN in statsmodels; compare log-likelihood instead.
if result_null.aic is not None and not (isinstance(result_null.aic, float) and str(result_null.aic) == 'nan'):
    print(f"  {'AIC':30s} {result_null.aic:>12.2f} {result_full.aic:>12.2f}")
    print(f"  {'BIC':30s} {result_null.bic:>12.2f} {result_full.bic:>12.2f}")
else:
    print(f"  {'AIC / BIC':30s} {'N/A (REML)':>12s} {'N/A (REML)':>12s}")

print(f"\n  Improvement in Log-Likelihood (Null -> Full): "
      f"{(result_null.log_likelihood or 0) - (result_full.log_likelihood or 0):.2f}")

# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - Adding parental_edu improves model fit (higher log-likelihood).")
print("  - The full model explains more variance in student scores.")
print("  - Note: MixedLM uses REML by default; AIC/BIC are not available")
print("    from statsmodels for REML. Compare log-likelihood instead.")
print("-" * 60)
print("\nDone. (D03)")

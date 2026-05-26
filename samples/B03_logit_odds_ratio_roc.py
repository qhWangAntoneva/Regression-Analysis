"""
B03: Logit with Odds Ratio Plot and ROC Curve
===============================================

Learning objectives:
  - Visualize odds ratios with 95% CI via odds_ratio_plot().
  - Generate and view an ROC curve with AUC.
  - Understand the relationship between model coefficients and
    predictive discrimination.

Generates synthetic binary data with 3 predictors (different DGP from B01).
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from src.modeling.specification import ModelSpec
from src.modeling.fitter import ModelFitter
from src.visualization.logit_plots import odds_ratio_plot, roc_curve_plot

# ---------------------------------------------------------------------------
# 1. Synthetic data (different DGP from B01)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=2027)
n = 600

credit_score = rng.normal(650, 80, n).clip(300, 850)
loan_amount = rng.uniform(5, 100, n)      # loan amount in thousands
years_employed = rng.uniform(0, 30, n)

# True DGP: logit(P_default) = 2 - 0.008*credit_score + 0.02*loan_amount - 0.1*years_employed
log_odds = 2.0 - 0.008 * credit_score + 0.02 * loan_amount - 0.1 * years_employed
prob = 1.0 / (1.0 + np.exp(-log_odds))
default = rng.binomial(1, prob)

data = pd.DataFrame({
    "default": default,
    "credit_score": credit_score.round(0).astype(int),
    "loan_amount": loan_amount.round(1),
    "years_employed": years_employed.round(1),
})

print("=" * 60)
print("  Loan Default Data (Logit DGP)")
print("=" * 60)
print(f"  N = {len(data)}")
print(f"  default=1: {default.sum()} ({default.mean()*100:.1f}%)")
print()

# ---------------------------------------------------------------------------
# 2. Fit logit model
# ---------------------------------------------------------------------------
spec = ModelSpec(
    dep_var="default",
    indep_vars=["credit_score", "loan_amount", "years_employed"],
    model_type="logit",
)

fitter = ModelFitter()
result = fitter.fit(spec, data, alpha=0.05)

print(result.summary())
print()

# ---------------------------------------------------------------------------
# 3. Coefficient table
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Coefficient Table")
print("=" * 60)
print(result.to_dataframe().to_string())
print()

# ---------------------------------------------------------------------------
# 4. Odds Ratio Forest Plot
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Odds Ratio Forest Plot (saved to odds_ratio_plot.html)")
print("=" * 60)
fig_or = odds_ratio_plot(result)
fig_or.write_html("odds_ratio_plot.html")
print("  File: odds_ratio_plot.html")
print()

# ---------------------------------------------------------------------------
# 5. ROC Curve
# ---------------------------------------------------------------------------
# Get predicted probabilities from the raw model
y_pred_prob = result._raw_model.predict()
y_true = data["default"].values

print("=" * 60)
print("  ROC Curve (saved to roc_curve.html)")
print("=" * 60)
fig_roc = roc_curve_plot(y_true, y_pred_prob)
fig_roc.write_html("roc_curve.html")
print("  File: roc_curve.html")
print()

# ---------------------------------------------------------------------------
# 6. Interpretation
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Interpretation")
print("=" * 60)
for c in result.coefficients:
    if c.name.lower() == "intercept":
        continue
    or_val = np.exp(c.coef)
    print(f"  {c.name:<20} OR = {or_val:.4f}  (95% CI: [{np.exp(c.ci_lower):.4f}, {np.exp(c.ci_upper):.4f}])")
    print(f"  {' ':<20} p = {c.pvalue:.4f} {c.significance}")

print()
print("  Odds ratio interpretation:")
print("  - OR > 1: predictor increases the odds of default = 1.")
print("  - OR < 1: predictor decreases the odds of default = 1.")
print("  - OR = 1: no effect.")
print()
print("  ROC AUC measures the model's ability to discriminate")
print("  between the two outcomes. AUC > 0.7 is considered acceptable,")
print("  AUC > 0.8 is excellent.")
print()

print("Done. Open the .html files in a browser to view the interactive plots.")

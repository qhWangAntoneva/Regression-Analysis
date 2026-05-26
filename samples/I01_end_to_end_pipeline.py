"""
I01: End-to-End Pipeline (Housing Data)
=========================================
Complete regression pipeline:
  1. Load housing data
  2. Check missing values, handle them
  3. Check for outliers
  4. Fit OLS model with log(sqft) transform
  5. Diagnostics: VIF, residual tests
  6. Coefficient plot
  7. Print full summary
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

from src.modeling.diagnostics import influence_stats, residual_tests, vif
from src.modeling.fitter import ModelFitter
from src.modeling.specification import ModelSpec
from src.preprocessing.missing import MissingValueHandler
from src.preprocessing.outliers import OutlierDetector
from src.utils.sample_data import load_housing_data
from src.visualization.coefficient import coefficient_plot_single

print("=" * 60)
print("  I01: End-to-End Regression Pipeline")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
print("\n[Step 1] Loading housing data...")
raw_data = load_housing_data()
print(f"  Shape: {raw_data.shape}")
print(f"  Columns: {list(raw_data.columns)}")

# ---------------------------------------------------------------------------
# Step 2: Missing values
# ---------------------------------------------------------------------------
print("\n[Step 2] Checking for missing values...")
handler = MissingValueHandler()
analysis = handler.analyze(raw_data)

print(f"  Total missing cells: {analysis['total_missing']}")
for col_name, col_info in analysis["columns"].items():
    if col_info["count"] > 0:
        print(f"    {col_name}: {col_info['count']} missing ({col_info['percentage']:.2f}%)")

# Impute mean for age
data_clean = handler.handle(raw_data, strategy="mean", columns=["age"])
print(f"  After imputation: {data_clean.shape[0]} rows, {int(data_clean['age'].isna().sum())} NaNs")

# ---------------------------------------------------------------------------
# Step 3: Outlier detection
# ---------------------------------------------------------------------------
print("\n[Step 3] Detecting outliers (IQR method)...")
detector = OutlierDetector()
data_flagged, outlier_summary = detector.flag_outliers(
    data_clean, columns=["sqft", "price", "age", "location_score"], method="iqr"
)
for col, info in outlier_summary.items():
    print(f"  {col}: {info['n_outliers']} outliers ({info['percentage']:.2f}%)")

# ---------------------------------------------------------------------------
# Step 4: Fit OLS with log(sqft) transform
# ---------------------------------------------------------------------------
print("\n[Step 4] Fitting OLS model with log(sqft) transform...")

spec = ModelSpec(
    dep_var="price",
    indep_vars=["sqft"],
    control_vars=["bedrooms", "age", "location_score", "floor", "has_garage"],
    transforms={"sqft": "log"},
    missing_strategy="drop",
)

fitter = ModelFitter()
result = fitter.fit(spec, data_clean)

print(result.summary())

# ---------------------------------------------------------------------------
# Step 5: Diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  [Step 5] Diagnostics")
print("=" * 60)

# 5a. VIF — use original variable names (the transformer creates sqft_log
# inside the fitter, so the pre-transform data_clean doesn't have it yet).
print("\n  Variance Inflation Factor (VIF) [using original untransformed vars]:")
try:
    vif_spec = ModelSpec(
        dep_var="price",
        indep_vars=["sqft", "bedrooms", "age", "location_score", "floor", "has_garage"],
    )
    vif_df = vif(data_clean, vif_spec)
    print(f"  {'Variable':<25s} {'VIF':>10s} {'Diagnosis':>12s}")
    print(f"  {'-'*47}")
    for _, row in vif_df.iterrows():
        print(f"  {str(row['variable']):<25s} {row['vif']:>10.4f} {str(row['diagnosis']):>12s}")
except Exception as e:
    print(f"  VIF computation failed: {e}")

# 5b. Residual tests
print("\n  Residual Diagnostic Tests:")
if result._raw_model is not None:
    residuals = result._raw_model.resid
    diag = residual_tests(residuals)

    print("    Shapiro-Wilk normality test:")
    print(f"      Statistic: {diag['shapiro_stat']:.4f}")
    print(f"      p-value:   {diag['shapiro_pvalue']:.6f}")
    print(f"      Normal?    {diag['shapiro_normal']}")

    print("    Durbin-Watson autocorrelation test:")
    print(f"      Statistic:  {diag['dw_stat']:.4f}")
    print(f"      Diagnosis:  {diag['dw_autocorrelation']}")

# 5c. Influence stats
print("\n  Influence Statistics (top 5 by Cook's distance):")
try:
    if result._raw_model is not None:
        inf_df = influence_stats(result._raw_model)
        top5 = inf_df.sort_values("cooks_d", ascending=False).head(5)
        print("  {:>6s} {:>12s} {:>12s}".format("Obs", "Cook's D", "Leverage"))
        print(f"  {'-'*30}")
        for _, row in top5.iterrows():
            obs = int(row['observation'])
            print(f"  {obs:>6d} {row['cooks_d']:>11.6f} {row['leverage']:>11.6f}")
except Exception as e:
    print(f"  Influence computation skipped: {e}")

# ---------------------------------------------------------------------------
# Step 6: Coefficient plot
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  [Step 6] Coefficient Plot")
print("=" * 60)

try:
    fig = coefficient_plot_single(result)
    # Save as HTML for viewing in browser
    fig.write_html("samples/I01_coefficient_plot.html")
    print("  Coefficient plot saved to: samples/I01_coefficient_plot.html")
except ImportError:
    print("  (Skipped — plotly not installed. Install with 'uv pip install plotly')")
except Exception as e:
    print(f"  (Skipped — plot generation error: {e})")

# ---------------------------------------------------------------------------
# Step 7: Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  [Step 7] Pipeline Summary")
print("=" * 60)
print(f"  Data loaded:       {raw_data.shape[0]} rows")
print("  Missing handled:   mean imputation for 'age'")
print(f"  Model type:        {result.model_type}")
print(f"  Transforms:        {result.transforms_applied}")
print(f"  R-squared:         {result.r_squared:.4f}")
print(f"  Adj. R-squared:    {result.adj_r_squared:.4f}")
print(f"  AIC:               {result.aic:.2f}")
print(f"  BIC:               {result.bic:.2f}")
print(f"  Observations used: {result.n_obs}")

print("\n" + "-" * 60)
print("  Pipeline complete. All steps executed successfully.")
print("-" * 60)
print("\nDone. (I01)")

"""
F03: Outlier Detection
=======================
Demonstrate OutlierDetector on housing data using both IQR and Z-score methods.
Flag outliers in key numeric columns and report counts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.preprocessing.outliers import OutlierDetector
from src.utils.sample_data import load_housing_data

print("=" * 60)
print("  F03: Outlier Detection")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_housing_data()
print(f"\nLoaded housing data: {data.shape[0]} rows")
print()

# ---------------------------------------------------------------------------
# 2. IQR method
# ---------------------------------------------------------------------------
detector = OutlierDetector()

numeric_cols = ["sqft", "age", "location_score", "price"]
data_iqr, summary_iqr = detector.flag_outliers(data, columns=numeric_cols, method="iqr")

print("Outlier Detection — IQR Method (multiplier=1.5):")
print(f"  {'Column':<20s} {'Outliers':>10s} {'Percentage':>12s}")
print(f"  {'-'*42}")
for col in numeric_cols:
    if col in summary_iqr:
        n_out = summary_iqr[col]["n_outliers"]
        pct = summary_iqr[col]["percentage"]
        print(f"  {col:<20s} {n_out:>10d} {pct:>11.2f}%")
    else:
        print(f"  {col:<20s} {'error':>10s}")

print()

# ---------------------------------------------------------------------------
# 3. Z-score method
# ---------------------------------------------------------------------------
data_z, summary_z = detector.flag_outliers(data, columns=numeric_cols, method="zscore")

print("Outlier Detection — Z-Score Method (threshold=3.0):")
print(f"  {'Column':<20s} {'Outliers':>10s} {'Percentage':>12s}")
print(f"  {'-'*42}")
for col in numeric_cols:
    if col in summary_z:
        n_out = summary_z[col]["n_outliers"]
        pct = summary_z[col]["percentage"]
        print(f"  {col:<20s} {n_out:>10d} {pct:>11.2f}%")
    else:
        print(f"  {col:<20s} {'error':>10s}")

# ---------------------------------------------------------------------------
# 4. Compare methods
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Method Comparison")
print("=" * 60)
print(f"  {'Column':<20s} {'IQR Outliers':>14s} {'Z-Score Outliers':>18s}")
print(f"  {'-'*52}")
for col in numeric_cols:
    iqr_n = summary_iqr.get(col, {}).get("n_outliers", 0)
    z_n = summary_z.get(col, {}).get("n_outliers", 0)
    print(f"  {col:<20s} {iqr_n:>14d} {z_n:>18d}")

# ---------------------------------------------------------------------------
# 5. Interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - IQR flags points beyond Q1-1.5*IQR or Q3+1.5*IQR (~0.7% expected")
print("    under normality for each tail).")
print("  - Z-score flags |z| > 3 (~0.3% expected under normality).")
print("  - IQR is more robust to non-normal distributions.")
print("  - Outliers in 'price' and 'sqft' may merit case-by-case review.")
print("-" * 60)
print("\nDone. (F03)")

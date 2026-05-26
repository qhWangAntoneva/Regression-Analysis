"""
F02: Missing Value Handling
============================
Demonstrate MissingValueHandler on the housing dataset (has NaN in age).
Analyze missing patterns, then apply mean-imputation and drop strategies.
Show before/after row counts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from src.utils.sample_data import load_housing_data
from src.preprocessing.missing import MissingValueHandler

print("=" * 60)
print("  F02: Missing Value Handling")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_housing_data()
print(f"\nOriginal data: {data.shape[0]} rows x {data.shape[1]} columns")
print()

# ---------------------------------------------------------------------------
# 2. Analyze missing values
# ---------------------------------------------------------------------------
handler = MissingValueHandler()
analysis = handler.analyze(data)

print("Missing Value Analysis:")
print(f"  Total rows:       {analysis['total_rows']}")
print(f"  Total columns:    {analysis['total_columns']}")
print(f"  Total missing:    {analysis['total_missing']}")
print()
print("  Per-column detail:")
for col_name, col_info in analysis["columns"].items():
    if col_info["count"] > 0:
        warn_flag = "  [WARN]" if col_info["warn"] else ""
        crit_flag = "  [CRITICAL]" if col_info["critical"] else ""
        print(f"    {col_name:<20s}  missing={col_info['count']:>3d}  "
              f"({col_info['percentage']:>5.2f}%){warn_flag}{crit_flag}")

# age column will have ~8 missing values
print()

# ---------------------------------------------------------------------------
# 3. Handle: mean imputation
# ---------------------------------------------------------------------------
data_mean = handler.handle(data, strategy="mean", columns=["age"])
n_missing_mean = int(data_mean["age"].isna().sum())
n_rows_mean = data_mean.shape[0]

print(f"Mean imputation -> {n_rows_mean} rows, age NaN count = {n_missing_mean}")

# ---------------------------------------------------------------------------
# 4. Handle: drop
# ---------------------------------------------------------------------------
data_dropped = handler.handle(data, strategy="drop")
n_rows_dropped = data_dropped.shape[0]

print(f"Drop strategy   -> {n_rows_dropped} rows")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print(f"\n  {'Strategy':<20s} {'Rows':>8s} {'Age NaN':>10s}")
print(f"  {'Original':<20s} {data.shape[0]:>8d} {int(data['age'].isna().sum()):>10d}")
print(f"  {'Mean imputation':<20s} {data_mean.shape[0]:>8d} {int(data_mean['age'].isna().sum()):>10d}")
print(f"  {'Drop':<20s} {data_dropped.shape[0]:>8d} {'N/A':>10s}")

# ---------------------------------------------------------------------------
# 6. Interpretation
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("  Interpretation:")
print("  - The housing dataset has a small number of NaN values in 'age'.")
print("  - Mean imputation preserves all rows but biases the distribution.")
print("  - Drop (listwise deletion) loses a few observations but is safer.")
print("  - With <2% missing, both strategies produce similar results.")
print("-" * 60)
print("\nDone. (F02)")

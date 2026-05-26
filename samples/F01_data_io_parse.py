"""
F01: FileParser — CSV Import & Data Summary

Demonstrates using FileParser to load a CSV file, preview the data,
and generate a data summary (column types, missing rates, memory usage).

We create a temporary CSV file with sample data and parse it via the
module's public API: `FileParser.parse()`, `preview_dataframe()`,
and `get_data_summary()`.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src import ...` works
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import tempfile

import numpy as np
import pandas as pd

from src.data_io.parser import FileParser, get_data_summary, preview_dataframe

# ── Create a temporary CSV file ──────────────────────────────────────────
rng = np.random.default_rng(20241121)
N = 100

tmp_data = pd.DataFrame({
    "id": range(1, N + 1),
    "age": rng.uniform(18, 65, N).round(1),
    "salary": rng.normal(50000, 15000, N).round(0).astype(int),
    "department": rng.choice(["Sales", "IT", "HR", "Finance"], N),
    "score": rng.uniform(0, 100, N).round(2),
})

# Inject a few missing values
tmp_data.loc[rng.choice(N, size=5, replace=False), "age"] = None
tmp_data.loc[rng.choice(N, size=3, replace=False), "score"] = None

tmp_path = os.path.join(tempfile.mkdtemp(), "sample_data.csv")
tmp_data.to_csv(tmp_path, index=False, encoding="utf-8")

print("=" * 60)
print("  F01: FileParser — CSV Import & Data Summary")
print("=" * 60)
print(f"\n  Temporary CSV: {tmp_path}")
print(f"  Rows written:  {len(tmp_data)}")
print(f"  Columns:       {list(tmp_data.columns)}")

# ── Parse ────────────────────────────────────────────────────────────────
parser = FileParser()
df = parser.parse(tmp_path)

print("\n" + "-" * 60)
print("  Parsed DataFrame Info")
print("-" * 60)
print(f"\n  Shape:            {df.shape}")
print(f"  Dtypes:\n{df.dtypes.to_string()}")

# ── Preview ──────────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Preview (first 5 rows via preview_dataframe)")
print("-" * 60)
preview = preview_dataframe(df, n=5)
print(preview.to_string())

# ── Data summary ─────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  Data Summary (via get_data_summary)")
print("-" * 60)
summary = get_data_summary(df)

print(f"\n  Rows:               {summary['n_rows']}")
print(f"  Columns:            {summary['n_cols']}")
print(f"  Memory:             {summary['memory_formatted']}")

print("\n  Column Types:")
for col, ctype in summary["column_types"].items():
    print(f"    {col:15s}  ->  {ctype}")

print("\n  Missing Rates:")
for col, rate in summary["missing_rates"].items():
    flag = " ***" if rate > 0 else ""
    print(f"    {col:15s}  {rate:.2%}{flag}")

# ── Cleanup ──────────────────────────────────────────────────────────────
os.remove(tmp_path)

print("\n" + "-" * 60)
print("  Interpretation")
print("-" * 60)
print("  FileParser auto-detected the CSV format and encoding (UTF-8).")
print("  `preview_dataframe()` provides a quick look at the data head.")
print("  `get_data_summary()` reports column types, missing rates, and")
print("  memory footprint — useful for data quality assessment before modeling.")

print("\n" + "=" * 60)
print("  Done — F01 complete.")
print("=" * 60)

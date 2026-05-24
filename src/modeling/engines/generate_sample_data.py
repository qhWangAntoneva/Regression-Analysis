# encoding: utf-8
"""Generate sample_ols.csv with known coefficients for testing.

Generates 200 rows with:
    - y = 2 + 0.5*x1 - 0.3*x2 + 0.1*(x3_B) - 0.1*(x3_C) + noise
    - x1: Normal(0, 1)
    - x2: Uniform(0, 1)
    - x3: categorical A/B/C -> dummy encoded as x3_B, x3_C
    - x4: continuous with 5% missing
    - cat1: binary 0/1

Uses np.random.seed(42) for reproducibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)
N = 200

# True coefficients for the data generating process
# y = intercept + 0.5*x1 - 0.3*x2 + 0.1*(cat=B) - 0.1*(cat=C) + 0.2*cat1 + noise
TRUE_INTERCEPT = 2.0
TRUE_X1 = 0.5
TRUE_X2 = -0.3
TRUE_X3_B = 0.1
TRUE_X3_C = -0.1
TRUE_CAT1 = 0.2
TRUE_NOISE_STD = 0.5

# Generate predictors
x1 = RNG.normal(0, 1, N)
x2 = RNG.uniform(0, 1, N)
x3_categories = RNG.choice(["A", "B", "C"], N, p=[0.4, 0.3, 0.3])
x4 = RNG.normal(5, 2, N)
cat1 = RNG.binomial(1, 0.5, N)

# Dummy encoding for x3
x3_B = (x3_categories == "B").astype(float)
x3_C = (x3_categories == "C").astype(float)

# Generate error term
noise = RNG.normal(0, TRUE_NOISE_STD, N)

# Generate y
y = (
    TRUE_INTERCEPT
    + TRUE_X1 * x1
    + TRUE_X2 * x2
    + TRUE_X3_B * x3_B
    + TRUE_X3_C * x3_C
    + TRUE_CAT1 * cat1
    + noise
)

# Build DataFrame
df = pd.DataFrame(
    {
        "y": y,
        "x1": x1,
        "x2": x2,
        "x3": x3_categories,
        "x3_B": x3_B,
        "x3_C": x3_C,
        "x4": x4,
        "cat1": cat1,
    }
)

# Introduce 5% missing values in x4 (10 out of 200)
missing_idx = RNG.choice(N, size=10, replace=False)
df.loc[missing_idx, "x4"] = np.nan

# Save
output_path = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures" / "sample_ols.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Sample data saved to {output_path}")
print(f"Shape: {df.shape}")
print(f"Missing in x4: {df['x4'].isna().sum()} / {N}")
print(f"\nTrue coefficients for verification:")
print(f"  Intercept = {TRUE_INTERCEPT}")
print(f"  x1        = {TRUE_X1}")
print(f"  x2        = {TRUE_X2}")
print(f"  x3_B      = {TRUE_X3_B}")
print(f"  x3_C      = {TRUE_X3_C}")
print(f"  cat1      = {TRUE_CAT1}")
print(f"  noise std = {TRUE_NOISE_STD}")

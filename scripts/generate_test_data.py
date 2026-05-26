"""CLI script to generate a synthetic regression test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(
    n_rows: int = 100,
    n_cols: int = 8,
    seed: int = 42,
    missing_ratio: float = 0.05,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame suitable for regression modeling.

    Parameters
    ----------
    n_rows : int
        Number of observations (rows).
    n_cols : int
        Total number of columns (including id and target). Must be >= 4.
    seed : int
        Random seed for reproducibility.
    missing_ratio : float
        Fraction of missing values to inject into continuous columns.

    Returns
    -------
    pd.DataFrame
        Generated dataset.
    """
    if n_cols < 4:
        raise ValueError("n_cols must be at least 4 (id + 2 predictors + target).")

    rng = np.random.default_rng(seed=seed)

    n_continuous = n_cols - 3  # reserve id + 1 categorical + y
    n_categorical = 2  # two categorical columns

    df = pd.DataFrame({"id": np.arange(1, n_rows + 1)})

    # Continuous predictors
    locs = np.linspace(0, 100, n_continuous)
    scales = np.full(n_continuous, 10)
    for i, (loc, scale) in enumerate(zip(locs, scales), start=1):
        col_name = f"x{i}"
        data = rng.normal(loc=loc, scale=scale, size=n_rows)

        # Inject missing values
        if missing_ratio > 0:
            mask = rng.random(size=n_rows) < missing_ratio
            data = data.astype(float)
            data[mask] = np.nan

        df[col_name] = data

    # Categorical predictors
    df["cat1"] = rng.choice(["A", "B", "C"], size=n_rows, p=[0.4, 0.35, 0.25])
    df["cat2"] = rng.choice(["X", "Y"], size=n_rows, p=[0.6, 0.4])

    # Target variable
    n_predictors = n_continuous + n_categorical  # includes cat dummified  # noqa: F841
    coeffs = rng.uniform(low=-0.5, high=1.0, size=n_continuous + 1)  # +1 for cat1
    linear = (
        coeffs[0] * df["x1"]
        if n_continuous >= 1
        else 0
    )
    if n_continuous >= 2:
        linear += coeffs[1] * df["x2"]
    if n_continuous >= 3:
        linear += coeffs[2] * df["x3"]

    df["y"] = (
        5.0
        + linear
        + 0.3 * df["cat1"].map({"A": 1, "B": 0, "C": -1})
        + rng.normal(loc=0, scale=2.0, size=n_rows)
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic regression test dataset."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows (observations). Default: 100.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of columns. Default: 8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--missing",
        type=float,
        default=0.05,
        help="Missing value ratio in continuous columns. Default: 0.05.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. Default: regression_test_data.csv in CWD.",
    )
    args = parser.parse_args()

    df = build_dataset(
        n_rows=args.rows,
        n_cols=args.cols,
        seed=args.seed,
        missing_ratio=args.missing,
    )

    output_path = Path(args.output) if args.output else Path.cwd() / "regression_test_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Dataset saved: {output_path.resolve()}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing ratio: {args.missing:.0%}")


if __name__ == "__main__":
    main()

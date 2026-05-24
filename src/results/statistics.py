# encoding: utf-8
"""Descriptive statistics and correlation matrix functions.

Provides utilities for generating summary statistics and correlation
matrices from pandas DataFrames.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def descriptive_stats(
    data: pd.DataFrame,
    variables: List[str],
) -> pd.DataFrame:
    """Compute descriptive statistics for selected variables.

    For each variable, computes: count, mean, standard deviation, minimum,
    25th percentile, median, 75th percentile, maximum, number of missing
    values, and missing rate.

    Args:
        data: The input DataFrame.
        variables: List of column names to summarize.

    Returns:
        A DataFrame where each row is a variable and columns are the
        computed statistics. The index is the variable name.

    Raises:
        ValueError: If any variable is not found in the data.
    """
    missing_vars = [v for v in variables if v not in data.columns]
    if missing_vars:
        raise ValueError(f"Variables not found in data: {missing_vars}")

    rows: list[dict[str, object]] = []
    for var in variables:
        col = data[var]
        non_missing = col.dropna()
        n_missing = int(col.isna().sum())
        missing_rate = n_missing / len(col) if len(col) > 0 else 0.0

        if len(non_missing) > 0:
            # Attempt numeric stats; fall back to frequency for categorical
            try:
                row = {
                    "变量": var,
                    "观测数": len(non_missing),
                    "均值": float(np.mean(non_missing)),
                    "标准差": float(np.std(non_missing, ddof=1)),
                    "最小值": float(np.min(non_missing)),
                    "25%": float(np.percentile(non_missing, 25)),
                    "50%": float(np.percentile(non_missing, 50)),
                    "75%": float(np.percentile(non_missing, 75)),
                    "最大值": float(np.max(non_missing)),
                    "缺失值数": n_missing,
                    "缺失率": round(missing_rate, 4),
                }
            except (TypeError, ValueError):
                # Categorical / non-numeric data
                row = {
                    "变量": var,
                    "观测数": len(non_missing),
                    "均值": float("nan"),
                    "标准差": float("nan"),
                    "最小值": float("nan"),
                    "25%": float("nan"),
                    "50%": float("nan"),
                    "75%": float("nan"),
                    "最大值": float("nan"),
                    "缺失值数": n_missing,
                    "缺失率": round(missing_rate, 4),
                }
        else:
            row = {
                "变量": var,
                "观测数": 0,
                "均值": float("nan"),
                "标准差": float("nan"),
                "最小值": float("nan"),
                "25%": float("nan"),
                "50%": float("nan"),
                "75%": float("nan"),
                "最大值": float("nan"),
                "缺失值数": n_missing,
                "缺失率": 1.0,
            }

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df.set_index("变量")
    return result_df


def correlation_matrix(
    data: pd.DataFrame,
    variables: List[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute the correlation matrix for selected variables.

    Uses pairwise complete observations (i.e., correlation is computed
    using all non-missing pairs).

    Args:
        data: The input DataFrame.
        variables: List of column names to correlate.
        method: Correlation method; ``'pearson'`` (default), ``'spearman'``,
            or ``'kendall'``.

    Returns:
        A DataFrame with the correlation matrix (variables × variables),
        with values rounded to 4 decimal places.

    Raises:
        ValueError: If any variable is not found in the data or the
            correlation method is invalid.
    """
    valid_methods = {"pearson", "spearman", "kendall"}
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {valid_methods}."
        )

    missing_vars = [v for v in variables if v not in data.columns]
    if missing_vars:
        raise ValueError(f"Variables not found in data: {missing_vars}")

    # Select only numeric columns
    numeric_vars: List[str] = []
    for v in variables:
        if pd.api.types.is_numeric_dtype(data[v]):
            numeric_vars.append(v)
        else:
            # Skip non-numeric variables with a warning
            import warnings

            warnings.warn(
                f"Variable '{v}' is not numeric and will be excluded "
                f"from the correlation matrix."
            )

    if len(numeric_vars) < 2:
        raise ValueError(
            "Need at least 2 numeric variables to compute correlations."
        )

    corr_matrix = data[numeric_vars].corr(method=method)
    corr_matrix = corr_matrix.round(4)
    return corr_matrix

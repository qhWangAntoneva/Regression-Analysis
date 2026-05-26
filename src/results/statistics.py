"""Descriptive statistics and correlation matrix functions.

Provides utilities for generating summary statistics and correlation
matrices from pandas DataFrames.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def descriptive_stats(
    data: pd.DataFrame,
    variables: list[str],
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
    variables: list[str],
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
        A DataFrame with the correlation matrix (variables x variables),
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
    numeric_vars: list[str] = []
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


# ------------------------------------------------------------------
# Phase 2 enhancements
# ------------------------------------------------------------------


def anova_oneway(
    data: pd.DataFrame,
    dv: str,
    group: str,
) -> dict[str, Any]:
    """Perform a one-way ANOVA (analysis of variance) for group comparisons.

    Tests whether the means of ``dv`` differ significantly across groups
    defined by ``group``.

    Args:
        data: The input DataFrame.
        dv: Name of the dependent variable (continuous).
        group: Name of the grouping variable (categorical/binary).

    Returns:
        A dictionary with keys:
            - ``'f_statistic'``: F-statistic.
            - ``'p_value'``: p-value of the F-test.
            - ``'df_between'``: Between-group degrees of freedom.
            - ``'df_within'``: Within-group degrees of freedom.
            - ``'ss_between'``: Between-group sum of squares.
            - ``'ss_within'``: Within-group sum of squares.
            - ``'ss_total'``: Total sum of squares.
            - ``'group_means'``: Dictionary of group -> mean of dv.
            - ``'group_counts'``: Dictionary of group -> count.
            - ``'group_stds'``: Dictionary of group -> std of dv.

    Raises:
        ValueError: If variables are not found in data or data is insufficient.
    """
    if dv not in data.columns:
        raise ValueError(f"Dependent variable '{dv}' not found in data.")
    if group not in data.columns:
        raise ValueError(f"Grouping variable '{group}' not found in data.")

    # Drop rows with missing values in either column
    valid = data[[dv, group]].dropna()
    if len(valid) < 3:
        raise ValueError(
            f"At least 3 valid observations are needed for ANOVA; got {len(valid)}."
        )

    groups = valid.groupby(group)[dv]
    group_means: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    group_stds: dict[str, float] = {}

    for name, grp in groups:
        key = str(name)
        group_means[key] = float(grp.mean())
        group_counts[key] = int(len(grp))
        group_stds[key] = float(grp.std()) if len(grp) > 1 else 0.0

    # Perform ANOVA using scipy
    grouped_data = [grp.values for _, grp in groups]
    f_stat, p_val = scipy_stats.f_oneway(*grouped_data)

    # Compute sum of squares manually
    grand_mean = float(valid[dv].mean())
    ss_between = sum(
        count * (mean - grand_mean) ** 2
        for count, mean in zip(group_counts.values(), group_means.values())
    )
    ss_within = sum(
        sum((x - mean) ** 2 for x in valid[valid[group] == name][dv])
        for name, mean in zip(group_means.keys(), group_means.values())
    )
    ss_total = ss_between + ss_within

    k = len(groups)
    n = len(valid)
    df_between = k - 1
    df_within = n - k

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "df_between": df_between,
        "df_within": df_within,
        "ss_between": float(round(ss_between, 6)),
        "ss_within": float(round(ss_within, 6)),
        "ss_total": float(round(ss_total, 6)),
        "group_means": group_means,
        "group_counts": group_counts,
        "group_stds": group_stds,
    }


def freq_table(
    data: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """Generate a frequency table with counts, percentages, and cumulative percentages.

    Args:
        data: The input DataFrame.
        col: Column name to tabulate.

    Returns:
        A DataFrame with columns:
        [类别, 频数, 百分比(%), 累积百分比(%)]

    Raises:
        ValueError: If the column is not found in the data.
    """
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in data.")

    valid = data[col].dropna()
    n_total = len(valid)

    value_counts = valid.value_counts()
    freq = value_counts.values
    labels = value_counts.index.tolist()

    pcts = (freq / n_total * 100).round(2)
    cum_pcts = np.cumsum(pcts)

    rows: list[dict[str, object]] = []
    for i, label in enumerate(labels):
        rows.append(
            {
                "类别": str(label),
                "频数": int(freq[i]),
                "百分比(%)": pcts[i],
                "累积百分比(%)": round(float(cum_pcts[i]), 2),
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.index = range(1, len(result_df) + 1)
    return result_df

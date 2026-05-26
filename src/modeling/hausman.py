"""Hausman specification test for panel model selection.

The Hausman test compares Fixed Effects (FE) and Random Effects (RE)
estimates to determine whether RE is consistent.  Under the null hypothesis
(H0), both FE and RE are consistent, but RE is more efficient.  Under the
alternative (H1), FE is consistent but RE is not.  A significant test
(p < 0.05) indicates that RE is inconsistent, so FE should be preferred.

Formula (Wooldridge, 2010):

    H = (b_fe - b_re)' [Var(b_fe) - Var(b_re)]^{-1} (b_fe - b_re)

Under H0, H ~ chi2(k), where k is the number of common coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats


def hausman_test(
    fe_params: dict[str, float],
    re_params: dict[str, float],
    fe_cov: np.ndarray,
    re_cov: np.ndarray,
    common_vars: list[str],
) -> dict[str, Any]:
    """Perform the Hausman specification test for panel models.

    Compares FE and RE coefficient vectors using their covariance matrices.
    Only variables present in *both* models are included in the test.

    Args:
        fe_params: Coefficient dictionary for FE model
            (``{variable_name: estimate}``).
        re_params: Coefficient dictionary for RE model
            (``{variable_name: estimate}``).
        fe_cov: Covariance matrix of FE coefficients (k×k numpy array,
            ordered to match *common_vars*).
        re_cov: Covariance matrix of RE coefficients (k×k numpy array,
            ordered to match *common_vars*).
        common_vars: List of variable names common to both models.

    Returns:
        A dictionary with keys:
            - ``'statistic'``: The Hausman chi-squared statistic.
            - ``'p_value'``: The p-value (chi-squared test).
            - ``'df'``: Degrees of freedom (number of common coefficients).
            - ``'common_vars'``: List of variable names used in the test.
            - ``'recommendation'``: ``'FE'`` if p < 0.05, else ``'RE'``.

    Raises:
        ValueError: If there are fewer than 1 common variable.
    """
    k = len(common_vars)
    if k < 1:
        raise ValueError(
            "Hausman test requires at least one common coefficient "
            "between FE and RE models."
        )

    # Build coefficient vectors in common-variable order
    b_fe = np.array([fe_params[v] for v in common_vars])
    b_re = np.array([re_params[v] for v in common_vars])

    # Coefficient difference
    d = b_fe - b_re

    # Variance of the difference
    var_d = fe_cov - re_cov

    # Invert variance matrix (handle non-positive-definite via pinv)
    try:
        inv_var_d = np.linalg.inv(var_d)
    except np.linalg.LinAlgError:
        inv_var_d = np.linalg.pinv(var_d)

    # Chi-squared statistic
    h_stat = float(d @ inv_var_d @ d)
    if h_stat < 0:
        h_stat = 0.0

    p_val = float(1.0 - scipy_stats.chi2.cdf(h_stat, k))
    recommendation = "FE" if p_val < 0.05 else "RE"

    return {
        "statistic": round(h_stat, 6),
        "p_value": round(p_val, 6),
        "df": k,
        "common_vars": list(common_vars),
        "recommendation": recommendation,
    }


def run_hausman_from_results(
    fe_result: Any,
    re_result: Any,
) -> dict[str, Any] | None:
    """Run the Hausman test from two ``ModelResult`` objects.

    Extracts the common coefficients and their covariance matrices from
    the raw fitted models stored on each ``ModelResult`` (via the
    ``_raw_model`` attribute).

    Args:
        fe_result: A ``ModelResult`` with ``model_type='panel'`` and
            ``panel_type='Panel FE'``, which must carry ``_raw_model``.
        re_result: A ``ModelResult`` with ``model_type='panel'`` and
            ``panel_type='Panel RE'``, which must carry ``_raw_model``.

    Returns:
        A Hausman test result dict, or ``None`` if the raw models are
        unavailable or the test cannot be computed.
    """
    fe_model = getattr(fe_result, "_raw_model", None)
    re_model = getattr(re_result, "_raw_model", None)
    if fe_model is None or re_model is None:
        return None

    try:
        fe_params = dict(fe_model.params)
        re_params = dict(re_model.params)

        # Find common variable names (exclude Intercept)
        common = sorted(
            set(fe_params.keys()) & set(re_params.keys())
            - {"Intercept", "const"}
        )
        if len(common) < 1:
            return None

        fe_cov = np.asarray(fe_model.cov.loc[common, common])
        re_cov = np.asarray(re_model.cov.loc[common, common])

        return hausman_test(fe_params, re_params, fe_cov, re_cov, common)
    except Exception:
        return None

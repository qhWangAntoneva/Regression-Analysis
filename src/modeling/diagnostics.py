"""Model diagnostic functions.

Provides tools for assessing regression model assumptions and quality:
multicollinearity (VIF), residual diagnostics (normality, autocorrelation),
influence statistics (Cook's distance, leverage), and a summary dictionary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.modeling.specification import ModelSpec, build_design_matrix
from src.results.table import ModelResult


def vif(
    data: pd.DataFrame,
    spec: ModelSpec,
    use_patsy: bool = True,
) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for multicollinearity detection.

    VIF values > 10 (or > 5 in conservative settings) indicate problematic
    multicollinearity.

    Args:
        data: The dataset.
        spec: The model specification. Only predictor variables are used.
        use_patsy: If True, use patsy to build the design matrix (respects
            categorical encoding). If False, use the raw numeric variables
            directly.

    Returns:
        A DataFrame with columns ``['variable', 'vif']``, sorted by VIF
        descending, plus ``'vif_sqrt'`` (the square root of VIF) and a
        ``'diagnosis'`` column.

    Raises:
        ValueError: If the design matrix has fewer than 2 columns or
            no valid observations.
    """
    if use_patsy:
        X, _ = build_design_matrix(spec, data)
    else:
        predictors = spec.all_predictors
        X = data[predictors].dropna().copy()
        # Add constant for VIF computation
        X = add_constant(X)

    if X.shape[0] == 0:
        raise ValueError("No valid observations for VIF computation.")
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 columns (including constant) for VIF.")

    vif_values: list[float] = []
    variable_names: list[str] = []
    diagnosis: list[str] = []

    for i in range(X.shape[1]):
        col_name = str(X.columns[i])
        # Skip the constant column for naming but compute VIF for it
        v_val = float(variance_inflation_factor(X.values, i))
        vif_values.append(v_val)
        variable_names.append(col_name)
        if v_val > 10:
            diagnosis.append("High")
        elif v_val > 5:
            diagnosis.append("Moderate")
        else:
            diagnosis.append("Low")

    result_df = pd.DataFrame(
        {
            "variable": variable_names,
            "vif": [round(v, 4) for v in vif_values],
            "vif_sqrt": [round(np.sqrt(v), 4) for v in vif_values],
            "diagnosis": diagnosis,
        }
    )
    result_df = result_df.sort_values("vif", ascending=False).reset_index(
        drop=True
    )
    return result_df


def residual_tests(residuals: np.ndarray) -> dict[str, float | str]:
    """Run standard residual diagnostic tests.

    Tests performed:
        - Shapiro-Wilk test for normality.
        - Durbin-Watson test for autocorrelation.

    Args:
        residuals: An array of model residuals.

    Returns:
        A dictionary with keys:
            - ``'shapiro_stat'``, ``'shapiro_pvalue'``, ``'shapiro_normal'``
            - ``'dw_stat'``, ``'dw_autocorrelation'``
    """
    results: dict[str, float | str] = {}

    # Shapiro-Wilk normality test
    if len(residuals) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        results["shapiro_stat"] = float(round(shapiro_stat, 6))
        results["shapiro_pvalue"] = float(shapiro_p)
        results["shapiro_normal"] = "Yes" if shapiro_p > 0.05 else "No"
    else:
        results["shapiro_stat"] = float("nan")
        results["shapiro_pvalue"] = float("nan")
        results["shapiro_normal"] = "Insufficient data"

    # Durbin-Watson autocorrelation test
    if len(residuals) >= 2:
        dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))
        results["dw_stat"] = round(dw, 4)
        # DW ~ 2 means no autocorrelation; < 1 or > 3 is concerning
        if dw < 1.0:
            results["dw_autocorrelation"] = "Positive (strong)"
        elif dw > 3.0:
            results["dw_autocorrelation"] = "Negative (strong)"
        elif dw < 1.5:
            results["dw_autocorrelation"] = "Positive (mild)"
        elif dw > 2.5:
            results["dw_autocorrelation"] = "Negative (mild)"
        else:
            results["dw_autocorrelation"] = "None"
    else:
        results["dw_stat"] = float("nan")
        results["dw_autocorrelation"] = "Insufficient data"

    return results


def influence_stats(
    model_results_wrapper: object,
) -> pd.DataFrame:
    """Compute influence diagnostics from a fitted statsmodels model.

    Provides Cook's distance and leverage (hat values) for each observation.

    Args:
        model_results_wrapper: A fitted statsmodels ``RegressionResultsWrapper``.

    Returns:
        A DataFrame with columns:
            - ``'cooks_d'``: Cook's distance.
            - ``'leverage'``: Hat-matrix diagonal (leverage).
            - ``'observation'``: Observation index.
    """
    try:
        influence = model_results_wrapper.get_influence()
    except AttributeError as exc:
        raise TypeError(
            "The object does not appear to be a statsmodels results object; "
            "it has no get_influence() method."
        ) from exc

    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

    df = pd.DataFrame(
        {
            "observation": range(len(cooks_d)),
            "cooks_d": cooks_d,
            "leverage": leverage,
        }
    )
    return df


def model_summary(result: ModelResult) -> dict[str, object]:
    """Return a comprehensive dictionary of model statistics.

    This is useful for programmatic access to all model quality metrics
    and for building custom reports.

    Args:
        result: A ModelResult from a fitted model.

    Returns:
        A dictionary containing all available model-level statistics
        and coefficient details.
    """
    summary: dict[str, object] = {
        "model_type": result.model_type,
        "method": result.method,
        "dep_var": result.dep_var,
        "specification": result.specification,
        "n_obs": result.n_obs,
        "n_params": result.n_params,
        "df_resid": result.df_resid,
        "r_squared": result.r_squared,
        "adj_r_squared": result.adj_r_squared,
        "rmse": result.rmse,
        "aic": result.aic,
        "bic": result.bic,
    }

    if result.f_statistic is not None:
        summary["f_statistic"] = result.f_statistic[0]
        summary["f_pvalue"] = result.f_statistic[1]

    if result.log_likelihood is not None:
        summary["log_likelihood"] = result.log_likelihood

    # Coefficient summary
    coef_table = result.to_dataframe().reset_index()
    summary["coefficients"] = coef_table.to_dict(orient="records")

    return summary

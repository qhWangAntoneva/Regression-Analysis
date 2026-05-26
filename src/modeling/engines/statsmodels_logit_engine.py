"""Statsmodels Logit regression engine.

Provides an adapter that runs binary logistic regression via statsmodels
and converts the results into the unified ModelResult data structure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.api as sm

from src.modeling.specification import ModelSpec, build_design_matrix, build_variable_labels
from src.results.table import CoefficientRow, ModelResult


def run_logit(
    data: pd.DataFrame,
    spec: ModelSpec,
    alpha: float = 0.05,
) -> tuple[Any, dict[str, str]]:
    """Fit a binary logistic regression model using statsmodels.

    Args:
        data: The dataset as a pandas DataFrame.
        spec: The model specification.
        alpha: Significance level for confidence intervals (default 0.05).

    Returns:
        A tuple of ``(fitted_model, variable_labels)`` where
        ``variable_labels`` maps raw column names to human-readable labels.

    Raises:
        ValueError: If the model fails to fit or does not converge.
    """
    X, y = build_design_matrix(spec, data)  # noqa: N806
    labels = build_variable_labels(spec, list(X.columns))

    try:
        logit_model = sm.Logit(y, X)
        fitted = logit_model.fit(disp=False)
    except Exception as exc:
        raise ValueError(f"Logit model failed to fit: {exc}") from exc

    # Check convergence
    converged = getattr(fitted, "mle_retvals", {}).get("converged", True)
    if not converged:
        raise ValueError(
            "Logit model failed to converge. "
            "This may indicate perfect separation or multicollinearity. "
            "Consider removing problematic predictors or using regularization."
        )

    return fitted, labels


def extract_logit(
    fitted_model: Any,
    alpha: float = 0.05,
    dep_var: str = "",
    specification: str = "",
    variable_labels: dict[str, str] | None = None,
) -> ModelResult:
    """Extract logit regression results into a ModelResult.

    Args:
        fitted_model: A fitted statsmodels Logit Results object.
        alpha: Significance level for confidence intervals (default 0.05,
            yielding 95% CIs).
        dep_var: Name of the dependent variable.
        specification: String representation of the model formula.
        variable_labels: Optional mapping from raw column names to
            human-readable display labels.

    Returns:
        A ``ModelResult`` populated with logit-specific statistics.
    """
    # --- Coefficient-level data ---
    params = fitted_model.params
    bse = fitted_model.bse
    zvalues = fitted_model.tvalues  # statsmodels stores z as tvalues for Logit
    pvalues = fitted_model.pvalues
    conf_int = fitted_model.conf_int(alpha=alpha)

    coefficients: list[CoefficientRow] = []
    for var_name in params.index:
        coef_val = float(params[var_name])
        se_val = float(bse[var_name])
        z_val = float(zvalues[var_name])
        p_val = float(pvalues[var_name])
        ci_low = float(conf_int.loc[var_name, 0])
        ci_high = float(conf_int.loc[var_name, 1])

        coefficients.append(
            CoefficientRow(
                name=str(var_name),
                coef=coef_val,
                se=se_val,
                t_stat=z_val,  # For logit, t_stat field holds z-statistic
                pvalue=p_val,
                ci_lower=ci_low,
                ci_upper=ci_high,
            )
        )

    # --- Model-level statistics ---
    n_obs = int(fitted_model.nobs)
    n_params = int(fitted_model.df_model) + 1  # Logit: df_model + intercept
    df_resid = int(fitted_model.df_resid)

    # McFadden's pseudo R-squared
    ll_model = float(fitted_model.llf)
    ll_null = float(fitted_model.llnull) if hasattr(fitted_model, "llnull") else 0.0
    pseudo_r_squared = float(1.0 - ll_model / ll_null) if ll_null != 0 else None

    # Likelihood ratio test
    llr: float | None = None
    llr_pvalue: float | None = None
    if hasattr(fitted_model, "llr"):
        llr = float(fitted_model.llr)
    if hasattr(fitted_model, "llr_pvalue"):
        llr_pvalue = float(fitted_model.llr_pvalue)

    aic = float(fitted_model.aic) if hasattr(fitted_model, "aic") else 0.0
    bic = float(fitted_model.bic) if hasattr(fitted_model, "bic") else 0.0

    return ModelResult(
        model_type="logit",
        coefficients=coefficients,
        n_obs=n_obs,
        n_params=n_params,
        df_resid=df_resid,
        r_squared=None,           # OLS-only
        adj_r_squared=None,       # OLS-only
        f_statistic=None,         # OLS-only
        rmse=None,                # OLS-only
        pseudo_r_squared=pseudo_r_squared,
        log_likelihood=ll_model,
        aic=aic,
        bic=bic,
        llr=llr,
        llr_pvalue=llr_pvalue,
        dep_var=dep_var,
        specification=specification,
        method="Logit",
        variable_labels=variable_labels if variable_labels is not None else {},
    )

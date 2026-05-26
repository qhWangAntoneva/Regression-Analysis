"""Statsmodels OLS regression engine.

Provides an adapter that runs OLS via statsmodels and converts the results
into the unified ModelResult data structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper

from src.modeling.specification import ModelSpec, build_design_matrix, build_variable_labels
from src.results.table import CoefficientRow, ModelResult


def extract_statsmodels(
    model: RegressionResultsWrapper,
    model_type: str = "OLS",
    alpha: float = 0.05,
    dep_var: str = "",
    specification: str = "",
    variable_labels: dict[str, str] | None = None,
) -> ModelResult:
    """Extract results from a statsmodels OLS Results object.

    Args:
        model: A fitted statsmodels ``RegressionResultsWrapper`` (from
            ``sm.OLS.fit()``).
        model_type: Label for the model type (default ``'OLS'``).
        alpha: Significance level for confidence intervals (default 0.05,
            yielding 95% CIs).
        dep_var: Name of the dependent variable.
        specification: String representation of the model formula.

    Returns:
        A ``ModelResult`` populated with all available statistics.
    """
    # --- Coefficient-level data ---
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues
    conf_int = model.conf_int(alpha=alpha)

    coefficients: list[CoefficientRow] = []
    for var_name in params.index:
        coef_val = float(params[var_name])
        se_val = float(bse[var_name])
        t_val = float(tvalues[var_name])
        p_val = float(pvalues[var_name])
        ci_low = float(conf_int.loc[var_name, 0])
        ci_high = float(conf_int.loc[var_name, 1])

        coefficients.append(
            CoefficientRow(
                name=str(var_name),
                coef=coef_val,
                se=se_val,
                t_stat=t_val,
                pvalue=p_val,
                ci_lower=ci_low,
                ci_upper=ci_high,
            )
        )

    # --- Model-level statistics ---
    n_obs = int(model.nobs)
    n_params = int(model.df_model) + (1 if "Intercept" in params.index else 0)
    df_resid = int(model.df_resid)

    r_squared: float | None = float(model.rsquared)
    adj_r_squared: float | None = float(model.rsquared_adj)

    f_stat: tuple[float, float] | None = None
    if hasattr(model, "fvalue") and hasattr(model, "f_pvalue"):
        fv = float(model.fvalue)
        fp = float(model.f_pvalue)
        if not (np.isnan(fv) or np.isnan(fp)):
            f_stat = (fv, fp)

    log_likelihood: float | None = None
    if hasattr(model, "llf") and model.llf is not None:
        log_likelihood = float(model.llf)

    aic = float(model.aic) if hasattr(model, "aic") else 0.0
    bic = float(model.bic) if hasattr(model, "bic") else 0.0

    # RMSE: sqrt(MSE) = sqrt(SSR / df_resid)
    ssr = float(model.ssr)
    rmse = float(np.sqrt(ssr / df_resid)) if df_resid > 0 else 0.0

    return ModelResult(
        model_type=model_type,
        coefficients=coefficients,
        n_obs=n_obs,
        n_params=n_params,
        df_resid=df_resid,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        f_statistic=f_stat,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        rmse=rmse,
        dep_var=dep_var,
        specification=specification,
        method=model_type,
        variable_labels=variable_labels if variable_labels is not None else {},
    )


def run_ols(
    data: pd.DataFrame,
    spec: ModelSpec,
    alpha: float = 0.05,
    cov_type: str = "nonrobust",
) -> ModelResult:
    """Run an OLS regression and return unified results.

    This is the primary entry point for OLS estimation. It:
    1. Builds the design matrix from the specification.
    2. Fits the OLS model via statsmodels.
    3. Extracts results via ``extract_statsmodels()``.
    4. Returns a ``ModelResult``.

    Args:
        data: The dataset as a pandas DataFrame.
        spec: The model specification.
        alpha: Significance level for confidence intervals (default 0.05).
        cov_type: Standard error type.  Supported values:
            ``'nonrobust'`` (default, classic SE),
            ``'HC0'``, ``'HC1'``, ``'HC2'``, ``'HC3'`` (robust SE).

    Returns:
        A populated ``ModelResult``.

    Raises:
        ValueError: If the design matrix cannot be built or the model
            cannot be fitted.
    """
    X, y = build_design_matrix(spec, data)  # noqa: N806
    labels = build_variable_labels(spec, list(X.columns))

    formula_str = f"{spec.dep_var} ~ {' + '.join(spec.all_predictors)}"
    if not spec.has_intercept:
        formula_str += " - 1"

    try:
        ols_model = OLS(y, X)
        if cov_type and cov_type != "nonrobust":
            fitted: RegressionResultsWrapper = ols_model.fit(
                cov_type=cov_type
            )
        else:
            fitted = ols_model.fit()
    except Exception as exc:
        raise ValueError(f"OLS model failed to fit: {exc}") from exc

    # Build specification string
    preds_str = " + ".join(spec.all_predictors)
    if spec.transforms:
        transform_parts = [f"{t}({v})" for v, t in spec.transforms.items()]
        preds_str += "  [" + ", ".join(transform_parts) + "]"
    if spec.interaction_terms:
        inter_parts = [f"{v1}:{v2}" for v1, v2 in spec.interaction_terms]
        preds_str += "  {" + ", ".join(inter_parts) + "}"
    spec_str = f"{spec.dep_var} ~ {preds_str}"
    if not spec.has_intercept:
        spec_str += " - 1 (no intercept)"
    if cov_type and cov_type != "nonrobust":
        spec_str += f"  [SE: {cov_type}]"

    result = extract_statsmodels(
        model=fitted,
        model_type="OLS",
        alpha=alpha,
        dep_var=spec.dep_var,
        specification=spec_str,
        variable_labels=labels,
    )
    result.transforms_applied = dict(spec.transforms)
    result.interaction_terms_applied = list(spec.interaction_terms)
    result.se_type = cov_type if cov_type else "nonrobust"
    return result

"""Statsmodels Count regression engine (Poisson and NegativeBinomial).

Provides adapters that run Poisson and NegativeBinomial regression via
statsmodels GLM and convert the results into the unified ModelResult
data structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from src.modeling.specification import (
    ModelSpec,
    build_design_matrix,
    build_variable_labels,
)
from src.results.table import CoefficientRow, ModelResult


def _validate_count_dv(y: pd.Series, model_type: str) -> None:
    """Validate that the dependent variable is suitable for count regression.

    Raises ValueError if the DV contains negative values or non-integer values.
    """
    if (y < 0).any():
        raise ValueError(
            f"{model_type} regression requires a non-negative dependent variable. "
            f"Found {int((y < 0).sum())} negative value(s)."
        )
    # Check for non-integer values (allow small floating-point tolerance)
    is_integer = np.allclose(y, np.round(y), atol=1e-8)
    if not is_integer:
        raise ValueError(
            f"{model_type} regression requires an integer-valued dependent variable "
            "(count data). Found non-integer values."
        )


def run_count_model(
    data: pd.DataFrame,
    spec: ModelSpec,
    alpha: float = 0.05,
    cov_type: str = "nonrobust",
) -> tuple[Any, dict[str, str]]:
    """Fit a Poisson or NegativeBinomial regression model using statsmodels GLM.

    Args:
        data: The dataset as a pandas DataFrame.
        spec: The model specification. ``model_type`` must be ``"poisson"``
            or ``"negbin"``.
        alpha: Significance level for confidence intervals (default 0.05).

    Returns:
        A tuple of ``(fitted_model, variable_labels)`` where
        ``variable_labels`` maps raw column names to human-readable labels.

    Raises:
        ValueError: If the model fails to fit, does not converge, or if the
            dependent variable violates count-data requirements.
    """
    X, y = build_design_matrix(spec, data)  # noqa: N806
    labels = build_variable_labels(spec, list(X.columns))

    # Validate count-data DV requirements
    _validate_count_dv(y, spec.model_type)

    # Handle exposure variable (rate model offset)
    exposure_var: str | None = getattr(spec, "exposure_var", None)
    offset: np.ndarray | None = None
    exposure_name: str | None = None
    if exposure_var:
        if exposure_var not in data.columns:
            raise ValueError(
                f"Exposure variable '{exposure_var}' not found in data columns."
            )
        exposure_vals = data.loc[X.index, exposure_var].values.astype(float)
        if (exposure_vals <= 0).any():
            raise ValueError(
                f"Exposure variable '{exposure_var}' contains non-positive "
                f"values. Exposure must be strictly positive."
            )
        offset = np.log(exposure_vals)
        exposure_name = exposure_var

        # Select the GLM family based on model_type
    model_type = spec.model_type.lower()
    if model_type == "poisson":
        family = sm.families.Poisson()
        display_name = "Poisson"
    elif model_type == "negbin":
        family = sm.families.NegativeBinomial()
        display_name = "NegativeBinomial"
    else:
        raise ValueError(
            f"Unsupported count model type '{spec.model_type}'. "
            f"Expected 'poisson' or 'negbin'."
        )

    try:
        glm_kwargs: dict[str, object] = {"family": family}
        if offset is not None:
            glm_kwargs["offset"] = offset
        glm_model = sm.GLM(y, X, **glm_kwargs)
        fit_kwargs: dict[str, object] = {}
        if cov_type and cov_type != "nonrobust":
            fit_kwargs["cov_type"] = cov_type
        fitted = glm_model.fit(**fit_kwargs)
    except Exception as exc:
        raise ValueError(f"{display_name} model failed to fit: {exc}") from exc

    # Check convergence
    converged = getattr(fitted, "converged", True)
    if not converged:
        raise ValueError(
            f"{display_name} model failed to converge. "
            "This may indicate separation or poor scaling of predictors. "
            "Consider standardizing continuous variables."
        )

    # Stash exposure metadata for the extractor
    fitted._exposure_name = exposure_name  # type: ignore[attr-defined]

    return fitted, labels


def extract_count_model(
    fitted_model: Any,
    alpha: float = 0.05,
    dep_var: str = "",
    specification: str = "",
    variable_labels: dict[str, str] | None = None,
) -> ModelResult:
    """Extract count regression results into a ModelResult.

    Args:
        fitted_model: A fitted statsmodels GLM Results object (Poisson or NB).
        alpha: Significance level for confidence intervals (default 0.05,
            yielding 95% CIs).
        dep_var: Name of the dependent variable.
        specification: String representation of the model formula.
        variable_labels: Optional mapping from raw column names to
            human-readable display labels.

    Returns:
        A ``ModelResult`` populated with count-model-specific statistics.
    """
    # --- Detect model type from the GLM family ---
    family_class = type(fitted_model.family).__name__
    if "Poisson" in family_class:
        model_type = "poisson"
        method = "Poisson"
    elif "NegativeBinomial" in family_class or "NegBin" in family_class:
        model_type = "negbin"
        method = "NegativeBinomial"
    else:
        # Fallback: try to infer from the family name attribute
        family_name = getattr(fitted_model.family, "family", "")
        if "poisson" in family_name.lower():
            model_type = "poisson"
            method = "Poisson"
        elif "neg" in family_name.lower() or "binomial" in family_name.lower():
            model_type = "negbin"
            method = "NegativeBinomial"
        else:
            model_type = "poisson"
            method = "Count"

    # --- Coefficient-level data ---
    params = fitted_model.params
    bse = fitted_model.bse
    zvalues = fitted_model.tvalues  # GLM stores z-statistics as tvalues
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
                t_stat=z_val,  # t_stat field holds z-statistic for count models
                pvalue=p_val,
                ci_lower=ci_low,
                ci_upper=ci_high,
            )
        )

    # --- Model-level statistics ---
    n_obs = int(fitted_model.nobs)
    n_params = int(fitted_model.df_model) + 1  # df_model + intercept
    df_resid = int(fitted_model.df_resid)

    # Log-likelihood
    ll_model = float(fitted_model.llf)

    # Null log-likelihood (for pseudo R-squared)
    # GLMResults computes llnull by fitting an intercept-only model
    try:
        ll_null = float(fitted_model.llnull)
    except (AttributeError, Exception):
        ll_null = 0.0

    # McFadden's pseudo R-squared
    pseudo_r_squared: float | None = None
    if ll_null != 0 and not np.isnan(ll_null):
        pseudo_r_squared_val = float(1.0 - ll_model / ll_null)
        # Clamp to [0, 1] to handle edge cases
        pseudo_r_squared = max(0.0, min(1.0, pseudo_r_squared_val))

    # Likelihood ratio test: compute from deviance
    # LLR = null_deviance - deviance
    llr: float | None = None
    llr_pvalue: float | None = None

    # Try the built-in llr first (unlikely to be set for GLM, but check)
    if hasattr(fitted_model, "llr"):
        llr = float(fitted_model.llr)
    if hasattr(fitted_model, "llr_pvalue"):
        llr_pvalue = float(fitted_model.llr_pvalue)

    # Fallback: compute from deviance
    if llr is None:
        deviance = float(fitted_model.deviance)
        null_deviance = None
        if hasattr(fitted_model, "null_deviance"):
            null_deviance = float(fitted_model.null_deviance)
        else:
            # Compute from Pearson chi2 if available
            try:
                null_deviance = float(fitted_model.pearson_chi2)
            except (AttributeError, Exception):
                null_deviance = deviance

        if null_deviance is not None and null_deviance > deviance:
            llr = float(null_deviance - deviance)
            df_llr = int(fitted_model.df_model)
            if llr > 0 and df_llr > 0:
                try:
                    llr_pvalue = float(1.0 - scipy_stats.chi2.cdf(llr, df_llr))
                except Exception:
                    llr_pvalue = None

    # Information criteria
    aic = float(fitted_model.aic) if hasattr(fitted_model, "aic") else 0.0
    bic = 0.0
    # Prefer LLF-based BIC (positive, for comparison); statsmodels GLM
    # deviance-based BIC may be negative and misleading.
    if hasattr(fitted_model, "bic_llf"):
        bic = float(fitted_model.bic_llf)
    elif hasattr(fitted_model, "bic"):
        bic = float(fitted_model.bic)
        # If deviance-based BIC is negative, compute LLF-based ourselves
        if bic < 0 and aic > 0:
            bic = float(fitted_model.bic_llf) if hasattr(fitted_model, "bic_llf") else aic + n_params * (np.log(n_obs) - 2)  # noqa: E501

    return ModelResult(
        model_type=model_type,
        coefficients=coefficients,
        n_obs=n_obs,
        n_params=n_params,
        df_resid=df_resid,
        r_squared=None,       # Not applicable for count models
        adj_r_squared=None,   # Not applicable for count models
        f_statistic=None,     # Not applicable for count models
        rmse=None,            # Not applicable for count models
        pseudo_r_squared=pseudo_r_squared,
        log_likelihood=ll_model,
        aic=aic,
        bic=bic,
        llr=llr,
        llr_pvalue=llr_pvalue,
        dep_var=dep_var,
        specification=specification,
        method=method,
        variable_labels=variable_labels if variable_labels is not None else {},
    )

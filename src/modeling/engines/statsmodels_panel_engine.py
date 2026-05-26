# encoding: utf-8
"""Panel data regression engine using linearmodels.

Provides adapters that run Fixed Effects (FE) and Random Effects (RE)
panel data models via linearmodels and convert the results into the
unified ModelResult data structure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from linearmodels.panel import PanelOLS, RandomEffects

from src.modeling.specification import ModelSpec, build_design_matrix, build_variable_labels
from src.results.table import CoefficientRow, ModelResult


def run_panel(
    data: pd.DataFrame,
    spec: ModelSpec,
    alpha: float = 0.05,
    cov_type: str = "clustered",
) -> Tuple[Any, Dict[str, str]]:
    """Fit a panel data model using linearmodels.

    Args:
        data: The dataset as a pandas DataFrame.
        spec: The model specification. Must have ``entity_var``,
            ``time_var``, and ``panel_model`` attributes (via duck
            typing or monkey-patching on the ModelSpec instance).
        alpha: Significance level for confidence intervals (default 0.05).
        cov_type: Standard error type. Default ``'clustered'`` uses
            entity-clustered standard errors.

    Returns:
        A tuple of ``(fitted_model, variable_labels)`` where
        ``variable_labels`` maps raw column names to human-readable labels.

    Raises:
        ValueError: If entity_var or time_var is missing, if there is
            only one entity, or if the model fails to fit.
    """
    entity_var: Optional[str] = getattr(spec, "entity_var", None)
    time_var: Optional[str] = getattr(spec, "time_var", None)
    panel_model: str = getattr(spec, "panel_model", "fixed")

    if not entity_var:
        raise ValueError("entity_var is required for panel data models.")
    if not time_var:
        raise ValueError("time_var is required for panel data models.")
    if panel_model not in ("fixed", "random"):
        raise ValueError(
            f"Unknown panel model '{panel_model}'. Must be 'fixed' or 'random'."
        )

    if entity_var not in data.columns:
        raise ValueError(
            f"Entity variable '{entity_var}' not found in data columns."
        )
    if time_var not in data.columns:
        raise ValueError(
            f"Time variable '{time_var}' not found in data columns."
        )

    # Build design matrix (handles missing values, transforms, intercept)
    X, y = build_design_matrix(spec, data)
    labels = build_variable_labels(spec, list(X.columns))

    # Align entity/time columns with the design matrix rows.
    # build_design_matrix may have dropped rows, so we use X.index
    # to select the matching rows from the original data.
    valid_rows = X.index
    entity_vals = data.loc[valid_rows, entity_var].values
    time_vals = data.loc[valid_rows, time_var].values
    panel_idx = pd.MultiIndex.from_arrays(
        [entity_vals, time_vals], names=[entity_var, time_var]
    )

    X = X.copy()
    y = y.copy()
    X.index = panel_idx
    y.index = panel_idx

    # Validate: need at least 2 entities for panel estimation
    n_entities = X.index.get_level_values(0).nunique()
    if n_entities < 2:
        raise ValueError(
            f"Panel data requires at least 2 entities, found {n_entities}."
        )

    # Fit the appropriate panel model
    try:
        if panel_model == "fixed":
            model = PanelOLS(y, X, entity_effects=True)
        else:
            model = RandomEffects(y, X)

        if cov_type == "clustered":
            fitted = model.fit(cov_type="clustered", cluster_entity=True)
        elif cov_type and cov_type != "nonrobust":
            fitted = model.fit(cov_type=cov_type)
        else:
            fitted = model.fit()
    except Exception as exc:
        raise ValueError(f"Panel data model failed to fit: {exc}") from exc

    return fitted, labels


def extract_panel(
    fitted_model: Any,
    alpha: float = 0.05,
    dep_var: str = "",
    specification: str = "",
    variable_labels: Optional[Dict[str, str]] = None,
) -> ModelResult:
    """Extract panel regression results into a ModelResult.

    Args:
        fitted_model: A fitted linearmodels panel model (PanelOLSResults
            or RandomEffectsResults object).
        alpha: Significance level for confidence intervals (default 0.05).
        dep_var: Name of the dependent variable.
        specification: String representation of the model formula.
        variable_labels: Optional mapping from raw column names to
            human-readable display labels.

    Returns:
        A ``ModelResult`` with ``model_type="panel"``, populated with
        within/between/overall R-squared, entity/time counts, and other
        panel-specific statistics.
    """
    params = fitted_model.params
    std_errors = fitted_model.std_errors
    t_stats = fitted_model.tstats
    pvalues = fitted_model.pvalues

    # Confidence intervals -- linearmodels API: PanelOLS accepts alpha kwarg,
    # RandomEffects does not (defaults to 95%).  Try alpha first, fall back.
    try:
        conf_int = fitted_model.conf_int(alpha=alpha)
    except TypeError:
        conf_int = fitted_model.conf_int()
    # In all tested versions the columns are named 'lower' and 'upper'.
    ci_lower_col = "lower"
    ci_upper_col = "upper"

    coefficients: list[CoefficientRow] = []
    for var_name in params.index:
        coefficients.append(
            CoefficientRow(
                name=str(var_name),
                coef=float(params[var_name]),
                se=float(std_errors[var_name]),
                t_stat=float(t_stats[var_name]),
                pvalue=float(pvalues[var_name]),
                ci_lower=float(conf_int.loc[var_name, ci_lower_col]),
                ci_upper=float(conf_int.loc[var_name, ci_upper_col]),
            )
        )

    # --- Model-level statistics -------------------------------------------------
    n_obs = int(fitted_model.nobs)
    has_intercept = "Intercept" in params.index or "const" in params.index

    # Detect FE vs RE
    is_fe = hasattr(fitted_model, "entity_effects")
    panel_method = "Panel FE" if is_fe else "Panel RE"

    # For FE: df_model already includes entity dummies, so it is the total
    # parameter count.  For RE: df_model counts only slope parameters;
    # add 1 for the implicit constant.
    if is_fe:
        n_params = int(fitted_model.df_model)
    else:
        n_params = int(fitted_model.df_model) + (0 if has_intercept else 1)

    df_resid = int(fitted_model.df_resid)

    # R-squared variants
    within_r2: Optional[float] = None
    between_r2: Optional[float] = None
    overall_r2: Optional[float] = None
    if hasattr(fitted_model, "rsquared_within"):
        within_r2 = float(fitted_model.rsquared_within)
    if hasattr(fitted_model, "rsquared_between"):
        between_r2 = float(fitted_model.rsquared_between)
    if hasattr(fitted_model, "rsquared_overall"):
        overall_r2 = float(fitted_model.rsquared_overall)

    # Primary R-squared: within for FE, overall for RE
    r_squared = within_r2 if within_r2 is not None else overall_r2

    # F-statistic (overall model significance)
    f_stat: Optional[Tuple[float, float]] = None
    if hasattr(fitted_model, "f_statistic") and fitted_model.f_statistic is not None:
        try:
            fs = fitted_model.f_statistic
            f_stat = (float(fs.stat), float(fs.pval))
        except Exception:
            pass

    # F-test for poolability (FE vs pooled OLS)
    f_pooled: Optional[Tuple[float, float]] = None
    if hasattr(fitted_model, "f_pooled") and fitted_model.f_pooled is not None:
        try:
            fp = fitted_model.f_pooled
            f_pooled = (float(fp.stat), float(fp.pval))
        except Exception:
            pass

    # Log-likelihood
    log_likelihood: Optional[float] = None
    try:
        if hasattr(fitted_model, "loglik") and fitted_model.loglik is not None:
            log_likelihood = float(fitted_model.loglik)
    except (TypeError, ValueError):
        pass

    # AIC / BIC (not always available for panel models)
    aic: float = 0.0
    bic: float = 0.0
    try:
        aic = float(fitted_model.aic) if hasattr(fitted_model, "aic") else 0.0
    except (TypeError, ValueError):
        pass
    try:
        bic = float(fitted_model.bic) if hasattr(fitted_model, "bic") else 0.0
    except (TypeError, ValueError):
        pass

    # RMSE from sum of squared residuals
    rmse: Optional[float] = None
    if hasattr(fitted_model, "resid_ss") and df_resid > 0:
        rmse = float(np.sqrt(fitted_model.resid_ss / df_resid))

    # Panel metadata: entity and time period counts
    n_entities: int = 0
    n_periods: int = 0
    try:
        n_entities = int(float(fitted_model.entity_info["total"]))
    except Exception:
        pass
    try:
        n_periods = int(float(fitted_model.time_info["total"]))
    except Exception:
        pass

    # Build the ModelResult
    result = ModelResult(
        model_type="panel",
        coefficients=coefficients,
        n_obs=n_obs,
        n_params=n_params,
        df_resid=df_resid,
        r_squared=r_squared,
        adj_r_squared=None,
        f_statistic=f_stat,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        rmse=rmse,
        dep_var=dep_var,
        specification=specification,
        method=panel_method,
        variable_labels=variable_labels if variable_labels is not None else {},
    )

    # Attach panel-specific fields via monkey-patching (these fields will
    # be formalised on ModelResult once the shared-file changes are applied).
    result.within_r_squared = within_r2           # type: ignore[attr-defined]
    result.between_r_squared = between_r2         # type: ignore[attr-defined]
    result.overall_r_squared = overall_r2          # type: ignore[attr-defined]
    result.entity_count = n_entities               # type: ignore[attr-defined]
    result.time_count = n_periods                  # type: ignore[attr-defined]
    result.panel_type = panel_method               # type: ignore[attr-defined]
    result.f_pooled = f_pooled                     # type: ignore[attr-defined]

    return result

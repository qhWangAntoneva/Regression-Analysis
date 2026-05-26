"""Statsmodels Mixed Linear Model (MixedLM) regression engine.

Provides an adapter that fits multilevel / mixed-effects models via
statsmodels ``MixedLM`` and converts the results into the unified
``ModelResult`` data structure.

MixedLM uses REML estimation by default.  It is more like OLS than like
Logit -- tests for fixed effects use z-statistics (statsmodels default),
and the output includes R-squared-like measures and RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper

from src.modeling.specification import ModelSpec, build_design_matrix, build_variable_labels
from src.results.table import CoefficientRow, ModelResult

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_mixedlm(
    data: pd.DataFrame,
    spec: ModelSpec,
    alpha: float = 0.05,
) -> tuple[MixedLMResultsWrapper, dict[str, str]]:
    """Fit a Mixed Linear Model.

    Args:
        data: The dataset as a pandas DataFrame.  Must include the column
            named by ``spec.group_var``.
        spec: The model specification.  **Must** carry a ``group_var``
            attribute naming the column used to define the clustering /
            grouping structure (e.g. school, hospital, country).
        alpha: Significance level for confidence intervals (default 0.05).

    Returns:
        A tuple ``(fitted_model, labels)`` where *fitted_model* is the
        raw ``MixedLMResultsWrapper`` returned by statsmodels and *labels*
        is the variable-label dictionary for the design-matrix columns.
        Group metadata (group_var name, group count) is stashed on the
        fitted model as ``_mixedlm_group_var`` and
        ``_mixedlm_group_count`` for downstream extraction.

    Raises:
        ValueError: If ``spec`` has no ``group_var``, the group column is
            missing from *data*, there are fewer than 2 unique groups, or
            the model fails to converge.
    """
    # --- group variable validation ---
    if not hasattr(spec, "group_var") or getattr(spec, "group_var") is None:
        raise ValueError(
            "MixedLM requires a group_var on the ModelSpec. "
            "Set spec.group_var to the name of the grouping column."
        )

    group_col: str = spec.group_var  # type: ignore[assignment]
    if group_col not in data.columns:
        raise ValueError(
            f"Group variable '{group_col}' not found in data columns: "
            f"{list(data.columns)}"
        )

    unique_groups = data[group_col].dropna().unique()
    group_count = len(unique_groups)
    if group_count < 2:
        raise ValueError(
            f"MixedLM requires at least 2 groups, but '{group_col}' has "
            f"only {group_count} unique value(s)."
        )

    # --- build design matrix ---
    X, y = build_design_matrix(spec, data)  # noqa: N806
    labels = build_variable_labels(spec, list(X.columns))

    # align the group vector with rows that survived missing-value removal
    groups = data.loc[X.index, group_col].values

    # --- fit ---
    try:
        model = sm.MixedLM(endog=y, exog=X, groups=groups)
        fitted: MixedLMResultsWrapper = model.fit(reml=True, disp=False)
    except Exception as exc:
        raise ValueError(f"MixedLM model failed to fit: {exc}") from exc

    # --- stash group metadata on the fitted object for extract_mixedlm ---
    fitted._mixedlm_group_var = group_col  # type: ignore[attr-defined]
    fitted._mixedlm_group_count = group_count  # type: ignore[attr-defined]

    return fitted, labels


def extract_mixedlm(
    fitted_model: MixedLMResultsWrapper,
    alpha: float = 0.05,
    dep_var: str = "",
    specification: str = "",
    variable_labels: dict[str, str] | None = None,
) -> ModelResult:
    """Extract MixedLM results into a ``ModelResult``.

    Args:
        fitted_model: A fitted ``MixedLMResultsWrapper`` (from
            ``run_mixedlm()`` or ``sm.MixedLM.fit()``).
        alpha: Significance level for confidence intervals (default 0.05).
        dep_var: Name of the dependent variable.
        specification: String representation of the model formula / spec.
        variable_labels: Optional human-readable label map for coefficient
            names.

    Returns:
        A ``ModelResult`` populated with fixed-effects estimates, random
        effects variance components, and model-level diagnostics.
    """
    # --- fixed effects ---
    fe_names = fitted_model.fe_params.index
    params = fitted_model.fe_params
    bse = fitted_model.bse_fe
    tvalues = fitted_model.tvalues.loc[fe_names]
    pvalues = fitted_model.pvalues.loc[fe_names]
    conf_int_full = fitted_model.conf_int(alpha=alpha)
    conf_int = conf_int_full.loc[fe_names]

    coefficients: list[CoefficientRow] = []
    for var_name in fe_names:
        coefficients.append(
            CoefficientRow(
                name=str(var_name),
                coef=float(params[var_name]),
                se=float(bse[var_name]),
                t_stat=float(tvalues[var_name]),
                pvalue=float(pvalues[var_name]),
                ci_lower=float(conf_int.loc[var_name, 0]),
                ci_upper=float(conf_int.loc[var_name, 1]),
            )
        )

    # --- model-level statistics ---
    n_obs = int(fitted_model.nobs)
    n_params = fitted_model.k_fe
    df_resid = int(fitted_model.df_resid)

    # compute conditional R² from fitted values
    y_endog = fitted_model.model.endog
    ss_resid = float(np.sum(fitted_model.resid ** 2))  # type: ignore[arg-type]
    ss_total = float(np.sum((y_endog - y_endog.mean()) ** 2))
    r_squared: float | None = None
    adj_r_squared: float | None = None
    if ss_total > 0:
        r_squared = 1.0 - ss_resid / ss_total
        if df_resid > 0:
            adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / df_resid

    # log-likelihood
    log_likelihood: float | None = None
    if fitted_model.llf is not None and not np.isnan(fitted_model.llf):
        log_likelihood = float(fitted_model.llf)

    # AIC / BIC -- valid only for ML; NaN for REML
    _aic = fitted_model.aic
    _bic = fitted_model.bic
    aic: float = float(_aic) if not (np.isnan(_aic) if isinstance(_aic, float) else False) else float("nan")  # noqa: E501
    bic: float = float(_bic) if not (np.isnan(_bic) if isinstance(_bic, float) else False) else float("nan")  # noqa: E501

    # RMSE
    rmse: float | None = (
        float(np.sqrt(ss_resid / df_resid)) if df_resid > 0 else None
    )

    # random effects variance components
    re_var: dict[str, float] = {}
    if fitted_model.cov_re is not None and fitted_model.cov_re.size > 0:
        for i, name in enumerate(fitted_model.cov_re.index):
            re_var[str(name)] = float(fitted_model.cov_re.iloc[i, i])

    # residual scale
    scale: float | None = None
    if hasattr(fitted_model, "scale") and fitted_model.scale is not None:
        scale = float(fitted_model.scale)

    # group metadata (stashed by run_mixedlm)
    group_var: str | None = getattr(fitted_model, "_mixedlm_group_var", None)
    group_count: int | None = getattr(fitted_model, "_mixedlm_group_count", None)
    if group_count is None and hasattr(fitted_model, "random_effects"):
        group_count = len(fitted_model.random_effects)

    result = ModelResult(
        model_type="mixedlm",
        coefficients=coefficients,
        n_obs=n_obs,
        n_params=n_params,
        df_resid=df_resid,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        f_statistic=None,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        rmse=rmse,
        dep_var=dep_var,
        specification=specification,
        method="MixedLM (REML)",
        variable_labels=variable_labels if variable_labels is not None else {},
    )

    # attach MixedLM-specific metadata dynamically
    # (these fields are documented for future addition to ModelResult in table.py)
    result.group_var = group_var  # type: ignore[attr-defined]
    result.re_var = re_var  # type: ignore[attr-defined]
    result.group_count = group_count  # type: ignore[attr-defined]
    result.mixedlm_scale = scale  # type: ignore[attr-defined]
    result.mixedlm_converged = bool(fitted_model.converged)  # type: ignore[attr-defined]

    return result


# ---------------------------------------------------------------------------
# Convenience: full run + extract
# ---------------------------------------------------------------------------


def run_and_extract_mixedlm(
    data: pd.DataFrame,
    spec: ModelSpec,
    alpha: float = 0.05,
) -> ModelResult:
    """Run MixedLM and extract results in one call.

    This is the primary high-level entry point.  It runs the full pipeline
    internally and returns a fully populated ``ModelResult`` with group
    metadata attached.

    Args:
        data: Dataset with a grouping column.
        spec: Model specification (must have ``group_var``).
        alpha: Significance level (default 0.05).

    Returns:
        A populated ``ModelResult``.
    """
    fitted, labels = run_mixedlm(data, spec, alpha=alpha)

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

    group_col = getattr(spec, "group_var", "unknown")
    group_count = getattr(fitted, "_mixedlm_group_count", 0)
    spec_str += f"  [groups: {group_col} ({group_count} levels)]"

    return extract_mixedlm(
        fitted_model=fitted,
        alpha=alpha,
        dep_var=spec.dep_var,
        specification=spec_str,
        variable_labels=labels,
    )

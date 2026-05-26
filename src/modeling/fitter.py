# encoding: utf-8
"""Model fitting dispatcher.

Provides a unified interface for fitting regression models, dispatching
to the appropriate engine based on the model type.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.modeling.engines.statsmodels_engine import run_ols
from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit
from src.modeling.specification import ModelSpec
from src.modeling.transforms import VariableTransformer
from src.results.table import ModelResult


class ModelFitter:
    """Dispatcher for fitting regression models.

    Supports OLS and logit regression. The ``fit()`` method routes to the
    appropriate engine based on ``spec.model_type`` (defaults to ``'ols'``).

    Usage::

        fitter = ModelFitter()
        result = fitter.fit(spec, data)
    """

    def __init__(self) -> None:
        """Initialize the fitter."""
        self._results: List[ModelResult] = []

    def fit(
        self,
        spec: ModelSpec,
        data: pd.DataFrame,
        alpha: float = 0.05,
        cov_type: str = "nonrobust",
        **kwargs: object,
    ) -> ModelResult:
        """Fit a single model specification to the data.

        Applies any variable transforms and interaction terms specified
        in the ``ModelSpec`` before fitting, then passes ``cov_type``
        to the engine for robust standard error support.

        Args:
            spec: The model specification.
            data: The dataset.
            alpha: Significance level for confidence intervals (default 0.05).
            cov_type: Standard error type (``'nonrobust'``, ``'HC0'``,
                ``'HC1'``, ``'HC2'``, ``'HC3'``).  Default ``'nonrobust'``.
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            A ModelResult containing the fitted model output, with
            ``transforms_applied``, ``interaction_terms_applied``, and
            ``se_type`` set appropriately.

        Raises:
            ValueError: If the model type is not supported.
        """
        # Build a copy of the spec with transformed column names substituted
        # so that patsy can find the new columns in the working data.
        transformer = VariableTransformer()
        working_data = data.copy()
        name_map: Dict[str, str] = {}  # original_name -> new_column_name

        if spec.transforms:
            working_data, meta = transformer.transform(
                working_data, spec.transforms
            )
            for var, tinfo in meta.items():
                ttype = spec.transforms[var]
                new_name = tinfo.get(ttype, "")
                if new_name:
                    name_map[var] = new_name

        # Build a new ModelSpec with the transformed column names
        if name_map:
            fit_indep = [
                name_map.get(v, v) for v in spec.indep_vars
            ]
            fit_control = [
                name_map.get(v, v) for v in spec.control_vars
            ]
            # Also update interaction terms to use transformed names
            fit_interactions = [
                (name_map.get(v1, v1), name_map.get(v2, v2))
                for v1, v2 in spec.interaction_terms
            ]
        else:
            fit_indep = list(spec.indep_vars)
            fit_control = list(spec.control_vars)
            fit_interactions = list(spec.interaction_terms)

        fit_spec = ModelSpec(
            dep_var=spec.dep_var,
            indep_vars=fit_indep,
            control_vars=fit_control,
            has_intercept=spec.has_intercept,
            transforms=dict(spec.transforms),
            interaction_terms=fit_interactions,
            missing_strategy=spec.missing_strategy,
            model_type=spec.model_type,
        )

        # Dispatch to the appropriate engine based on model type
        # Build specification string
        preds_str = " + ".join(fit_spec.all_predictors)
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

        if spec.model_type in ("logit", "probit", "poisson", "negbin"):
            fitted, var_labels = run_logit(working_data, fit_spec)
            result = extract_logit(
                fitted_model=fitted,
                alpha=alpha,
                dep_var=spec.dep_var,
                specification=spec_str,
                variable_labels=var_labels,
            )
        else:
            result = run_ols(
                working_data, fit_spec, alpha=alpha, cov_type=cov_type
            )

        # Copy metadata fields back from the fit_spec's result
        result.transforms_applied = dict(spec.transforms)
        result.interaction_terms_applied = list(spec.interaction_terms)
        result.se_type = cov_type if cov_type else "nonrobust"

        self._results.append(result)
        return result

    def fit_multiple(
        self,
        specs: List[ModelSpec],
        data: pd.DataFrame,
        alpha: float = 0.05,
        cov_type: str = "nonrobust",
        **kwargs: object,
    ) -> List[ModelResult]:
        """Fit multiple model specifications to the same data.

        This is useful for model comparison tables, e.g. adding control
        variables incrementally.

        Args:
            specs: A list of model specifications.
            data: The dataset.
            alpha: Significance level for confidence intervals (default 0.05).
            cov_type: Standard error type.
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            A list of ModelResult objects, one per input specification.
        """
        results: List[ModelResult] = []
        for spec in specs:
            result = self.fit(
                spec, data, alpha=alpha, cov_type=cov_type, **kwargs
            )
            results.append(result)
        return results

    @property
    def fitted_results(self) -> List[ModelResult]:
        """Return all results fitted by this instance."""
        return list(self._results)

    def clear(self) -> None:
        """Clear all cached results."""
        self._results.clear()

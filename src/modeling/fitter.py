# encoding: utf-8
"""Model fitting dispatcher.

Provides a unified interface for fitting regression models, dispatching
to the appropriate engine based on the model type.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.modeling.engines.statsmodels_engine import run_ols
from src.modeling.specification import ModelSpec
from src.results.table import ModelResult


class ModelFitter:
    """Dispatcher for fitting regression models.

    Currently supports only OLS. The ``fit()`` method routes to the
    appropriate engine based on the model type stored in the spec
    (defaults to OLS).

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
        **kwargs: object,
    ) -> ModelResult:
        """Fit a single model specification to the data.

        Args:
            spec: The model specification.
            data: The dataset.
            alpha: Significance level for confidence intervals (default 0.05).
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            A ModelResult containing the fitted model output.

        Raises:
            ValueError: If the model type is not supported.
        """
        result = run_ols(data, spec, alpha=alpha)
        self._results.append(result)
        return result

    def fit_multiple(
        self,
        specs: List[ModelSpec],
        data: pd.DataFrame,
        alpha: float = 0.05,
        **kwargs: object,
    ) -> List[ModelResult]:
        """Fit multiple model specifications to the same data.

        This is useful for model comparison tables, e.g. adding control
        variables incrementally.

        Args:
            specs: A list of model specifications.
            data: The dataset.
            alpha: Significance level for confidence intervals (default 0.05).
            **kwargs: Additional engine-specific keyword arguments.

        Returns:
            A list of ModelResult objects, one per input specification.
        """
        results: List[ModelResult] = []
        for spec in specs:
            result = self.fit(spec, data, alpha=alpha, **kwargs)
            results.append(result)
        return results

    @property
    def fitted_results(self) -> List[ModelResult]:
        """Return all results fitted by this instance."""
        return list(self._results)

    def clear(self) -> None:
        """Clear all cached results."""
        self._results.clear()

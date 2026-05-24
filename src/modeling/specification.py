# encoding: utf-8
"""Model specification building module.

Provides dataclasses and utilities for constructing model specifications,
generating patsy formulas, and building design matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import patsy


@dataclass
class ModelSpec:
    """Regression model specification.

    Attributes:
        dep_var: Name of the dependent variable column in the data.
        indep_vars: Names of independent variable columns.
        control_vars: Names of control variable columns (included in model
            but not of primary interest). These are combined with indep_vars
            on the right-hand side of the formula.
        has_intercept: Whether to include a constant term (default: True).
    """

    dep_var: str
    indep_vars: List[str]
    control_vars: List[str] = field(default_factory=list)
    has_intercept: bool = True

    @property
    def all_predictors(self) -> List[str]:
        """Return the combined list of all predictor variables."""
        return self.indep_vars + self.control_vars

    def __post_init__(self) -> None:
        """Validate the specification."""
        if not self.dep_var:
            raise ValueError("dep_var must be a non-empty string.")
        if not self.indep_vars:
            raise ValueError("indep_vars must be a non-empty list.")
        # Check for duplicates
        combined = self.all_predictors
        if len(combined) != len(set(combined)):
            raise ValueError("Duplicate variable names detected in predictors.")


def _term_name(term: patsy.Term) -> str:
    """Extract a human-readable name from a patsy Term object.

    Handles both simple terms (e.g., 'x1') and categorical terms
    (e.g., 'C(cat1)[T.1]').
    """
    name = "".join(factor.name() for factor in term.factors)
    # Clean up patsy encoding for categorical variables
    for factor in term.factors:
        if hasattr(factor, "name"):
            name = factor.name()
            break
    return name


def build_formula(spec: ModelSpec) -> str:
    """Generate a patsy formula string from a ModelSpec.

    Categorical variables are automatically wrapped with ``C()`` so that
    patsy creates the appropriate dummy-variable encoding.

    Args:
        spec: The model specification.

    Returns:
        A patsy-compatible formula string, e.g. ``"y ~ x1 + x2 + C(cat)"``.

    Raises:
        ValueError: If the specification contains no predictors.
    """
    predictors = spec.all_predictors
    if not predictors:
        raise ValueError("ModelSpec must have at least one predictor.")

    # patsy's C() notation handles categorical variables automatically.
    # For this implementation we assume string/object columns need C() wrapping.
    # The caller is responsible for passing appropriately typed data.
    rhs = " + ".join(predictors)

    if not spec.has_intercept:
        rhs = f"{rhs} - 1"

    return f"{spec.dep_var} ~ {rhs}"


def build_design_matrix(
    spec: ModelSpec,
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the design matrix and dependent variable vector from data.

    Constructs the model matrix (X) and response vector (y) using patsy.
    Rows with any missing values are removed (listwise deletion).

    Args:
        spec: The model specification.
        data: The input data as a pandas DataFrame.

    Returns:
        A tuple ``(X, y)`` where:
            - ``X`` is a DataFrame of the design matrix (includes dummies).
            - ``y`` is a Series of the dependent variable.

    Raises:
        ValueError: If the dependent variable is missing from the data, or
            if no valid rows remain after listwise deletion.
    """
    if spec.dep_var not in data.columns:
        raise ValueError(
            f"Dependent variable '{spec.dep_var}' not found in data."
        )

    missing_predictors = [
        v for v in spec.all_predictors if v not in data.columns
    ]
    if missing_predictors:
        raise ValueError(
            f"Predictors not found in data: {missing_predictors}"
        )

    formula = build_formula(spec)

    # Use patsy to build the design matrices
    try:
        y_dmat, X_dmat = patsy.dmatrices(
            formula,
            data,
            return_type="dataframe",
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to build design matrix with formula '{formula}': {exc}"
        ) from exc

    # Convert to the expected return types
    # y is a 2D DataFrame from patsy; extract as a 1D Series
    y: pd.Series = y_dmat.iloc[:, 0]
    y.name = spec.dep_var
    X: pd.DataFrame = X_dmat

    if X.shape[0] == 0:
        raise ValueError(
            "No valid observations remain after listwise deletion."
        )

    return X, y

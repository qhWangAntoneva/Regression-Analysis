# encoding: utf-8
"""Model specification building module.

Provides dataclasses and utilities for constructing model specifications,
generating patsy formulas, and building design matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import patsy

SUPPORTED_MODEL_TYPES = ("ols", "logit", "probit", "mixedlm", "panel", "poisson", "negbin")
"""All model types supported by the regression analysis framework."""


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
    transforms: Dict[str, str] = field(default_factory=dict)
    interaction_terms: List[Tuple[str, str]] = field(default_factory=list)
    missing_strategy: str = "drop"
    model_type: str = "ols"

    # MixedLM-specific
    group_var: Optional[str] = None
    """Grouping variable for MixedLM (multilevel) models."""

    # Panel-specific
    entity_var: Optional[str] = None
    """Entity (cross-section) identifier for panel data models."""
    time_var: Optional[str] = None
    """Time identifier for panel data models."""
    panel_model: Optional[str] = None
    """Panel estimator type: ``'fixed'`` or ``'random'``."""

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
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Must be one of {SUPPORTED_MODEL_TYPES}"
            )
        # Check for duplicates
        combined = self.all_predictors
        if len(combined) != len(set(combined)):
            raise ValueError("Duplicate variable names detected in predictors.")

        # Validate MixedLM: group_var is required at runtime, but not enforced at
        # construction time (engines validate and set group_var dynamically).
        if self.model_type == "mixedlm" and self.group_var is not None:
            if not isinstance(self.group_var, str) or not self.group_var.strip():
                raise ValueError("mixedlm group_var must be a non-empty string.")

        # Validate Panel: fields are validated at engine-call time.
        if self.model_type == "panel":
            if self.panel_model is not None and self.panel_model not in ("fixed", "random"):
                raise ValueError(
                    f"panel_model must be 'fixed' or 'random', got '{self.panel_model}'."
                )


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


def build_formula(
    spec: ModelSpec,
    use_transformed_names: bool = False,
    name_map: Optional[Dict[str, str]] = None,
) -> str:
    """Generate a patsy formula string from a ModelSpec.

    Categorical variables are automatically wrapped with ``C()`` so that
    patsy creates the appropriate dummy-variable encoding.

    When ``use_transformed_names`` is ``True`` and a ``name_map`` is
    provided, original variable names are replaced with their transformed
    column names in the formula string.
    Interaction terms are appended as ``var1:var2`` terms.

    Args:
        spec: The model specification.
        use_transformed_names: Whether to substitute variable names using
            the ``name_map``.
        name_map: Mapping of ``{original_variable: new_column_name}`` to
            use when ``use_transformed_names`` is ``True``.

    Returns:
        A patsy-compatible formula string, e.g. ``"y ~ x1 + x2 + C(cat)"``.

    Raises:
        ValueError: If the specification contains no predictors.
    """
    predictors = spec.all_predictors
    if not predictors:
        raise ValueError("ModelSpec must have at least one predictor.")

    # Substitute with transformed names if requested
    if use_transformed_names and name_map:
        resolved: List[str] = []
        for p in predictors:
            if p in name_map:
                resolved.append(name_map[p])
            else:
                resolved.append(p)
        rhs_parts = resolved
    else:
        rhs_parts = list(predictors)

    # Append interaction terms as "var1:var2"
    if spec.interaction_terms:
        for v1, v2 in spec.interaction_terms:
            # Use transformed names if available
            n1 = name_map.get(v1, v1) if name_map else v1
            n2 = name_map.get(v2, v2) if name_map else v2
            rhs_parts.append(f"{n1}:{n2}")

    rhs = " + ".join(rhs_parts)

    if not spec.has_intercept:
        rhs = f"{rhs} - 1"

    return f"{spec.dep_var} ~ {rhs}"


def build_design_matrix(
    spec: ModelSpec,
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the design matrix and dependent variable vector from data.

    Constructs the model matrix (X) and response vector (y) using patsy.
    Handles missing values according to ``spec.missing_strategy``.

    Args:
        spec: The model specification.
        data: The input data as a pandas DataFrame.

    Returns:
        A tuple ``(X, y)`` where:
            - ``X`` is a DataFrame of the design matrix (includes dummies).
            - ``y`` is a Series of the dependent variable.

    Raises:
        ValueError: If the dependent variable is missing from the data, or
            if no valid rows remain after deletion.
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

    # Apply missing strategy before building design matrix
    working_data = data.copy()
    cols_for_missing = [spec.dep_var] + spec.all_predictors
    missing_rows = working_data[cols_for_missing].isna().any(axis=1)

    if missing_rows.any():
        if spec.missing_strategy == "drop":
            working_data = working_data.loc[~missing_rows].copy()
        elif spec.missing_strategy == "mean":
            for col in spec.all_predictors:
                if working_data[col].isna().any():
                    working_data[col] = working_data[col].fillna(
                        working_data[col].mean()
                    )
        elif spec.missing_strategy == "median":
            for col in spec.all_predictors:
                if working_data[col].isna().any():
                    working_data[col] = working_data[col].fillna(
                        working_data[col].median()
                    )

    formula = build_formula(spec)

    # Use patsy to build the design matrices
    try:
        y_dmat, X_dmat = patsy.dmatrices(
            formula,
            working_data,
            return_type="dataframe",
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to build design matrix with formula '{formula}': {exc}"
        ) from exc

    # Convert to the expected return types
    y: pd.Series = y_dmat.iloc[:, 0]
    y.name = spec.dep_var
    X: pd.DataFrame = X_dmat

    if X.shape[0] == 0:
        raise ValueError(
            "No valid observations remain after listwise deletion."
        )

    return X, y


def build_variable_labels(
    spec: ModelSpec,
    X_columns: List[str],
) -> Dict[str, str]:
    """Generate human-readable labels for design matrix columns.

    Parses patsy-generated column names (e.g. ``cat[T.b]``) into
    user-friendly labels (e.g. ``cat: b``).  Interaction terms are
    split on ``:``, decoded part-by-part, and rejoined with `` × ``.

    Args:
        spec: The model specification (reserved for future use).
        X_columns: List of column names from the design matrix DataFrame.

    Returns:
        A dictionary mapping each raw column name to its display label.
    """
    import re

    # Matches both main-effect categoricals (cat[T.level]) and
    # interaction-only categoricals (cat[level], no "T." prefix).
    _CAT_PART = re.compile(r"^(\w+)\[T?\.?([^\]]+)\]$")

    labels: Dict[str, str] = {}
    for col in X_columns:
        if col == "Intercept":
            labels[col] = "Intercept"
            continue
        parts = col.split(":")
        decoded = []
        for part in parts:
            m = _CAT_PART.match(part)
            if m:
                decoded.append(f"{m.group(1)}: {m.group(2)}")
            else:
                decoded.append(part)
        labels[col] = " × ".join(decoded) if len(decoded) > 1 else decoded[0]

    return labels

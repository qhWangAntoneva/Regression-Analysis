"""Modeling module.

Provides model fitting, specification building, variable transformation,
diagnostic functions, and the Hausman specification test for panel models.
"""

from src.modeling.diagnostics import influence_stats, model_summary, residual_tests, vif
from src.modeling.fitter import ModelFitter
from src.modeling.hausman import hausman_test, run_hausman_from_results
from src.modeling.specification import (
    SUPPORTED_MODEL_TYPES,
    ModelSpec,
    build_design_matrix,
    build_formula,
    build_variable_labels,
)
from src.modeling.transforms import VariableTransformer

__all__ = [
    "ModelFitter",
    "ModelSpec",
    "VariableTransformer",
    "SUPPORTED_MODEL_TYPES",
    "build_formula",
    "build_design_matrix",
    "build_variable_labels",
    "vif",
    "residual_tests",
    "influence_stats",
    "model_summary",
    "hausman_test",
    "run_hausman_from_results",
]

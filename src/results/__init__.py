"""Results module.

Provides unified result data structures (ModelResult, CoefficientRow),
descriptive statistics, correlation matrices, ANOVA, frequency tables,
model comparison utilities, and human-readable summary generation.
"""

from src.results.statistics import anova_oneway, correlation_matrix, descriptive_stats, freq_table
from src.results.summary_generator import (
    generate_assumption_check_text,
    generate_coefficient_interpretation,
    generate_summary_text,
)
from src.results.table import CoefficientRow, ModelResult, compare_models

__all__ = [
    "ModelResult",
    "CoefficientRow",
    "compare_models",
    "descriptive_stats",
    "correlation_matrix",
    "anova_oneway",
    "freq_table",
    "generate_summary_text",
    "generate_coefficient_interpretation",
    "generate_assumption_check_text",
]

"""Preprocessing module.

Provides missing value handling, outlier detection, and variable type detection
for data preparation before regression analysis.
"""

from src.preprocessing.missing import MissingValueHandler
from src.preprocessing.outliers import OutlierDetector
from src.preprocessing.type_detector import VariableInfo, VariableTypeDetector

__all__ = [
    "MissingValueHandler",
    "OutlierDetector",
    "VariableInfo",
    "VariableTypeDetector",
]

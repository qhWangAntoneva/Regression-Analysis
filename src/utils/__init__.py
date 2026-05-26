"""Utilities module.

Provides custom exception classes, logging configuration, session persistence,
sample data loaders, and the pre-computed gallery of analysis scenarios.
"""

from src.utils.exceptions import (
    DataParseError,
    ExportError,
    ModelingError,
    RegressionAnalysisError,
    VisualizationError,
)
from src.utils.gallery import GalleryItem, get_gallery_index, get_gallery_item, get_gallery_items
from src.utils.logger import get_logger
from src.utils.persistence import (
    clear_session,
    load_session,
    save_session,
    session_cache_exists,
)
from src.utils.sample_data import (
    get_sample_datasets,
    load_air_quality_data,
    load_housing_data,
    load_sample_dataset,
    load_wages_data,
)

__all__ = [
    "RegressionAnalysisError",
    "DataParseError",
    "ModelingError",
    "VisualizationError",
    "ExportError",
    "get_logger",
    "save_session",
    "load_session",
    "clear_session",
    "session_cache_exists",
    "get_sample_datasets",
    "load_housing_data",
    "load_wages_data",
    "load_air_quality_data",
    "load_sample_dataset",
    "GalleryItem",
    "get_gallery_items",
    "get_gallery_index",
    "get_gallery_item",
]

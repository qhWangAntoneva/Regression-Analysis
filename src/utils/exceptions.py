"""Custom exception hierarchy for the regression analysis tool."""


class RegressionAnalysisError(Exception):
    """Base exception for all regression analysis tool errors."""
    pass


class DataParseError(RegressionAnalysisError):
    """Raised when data file parsing fails."""
    pass


class ModelingError(RegressionAnalysisError):
    """Raised when model fitting or specification fails."""
    pass


class VisualizationError(RegressionAnalysisError):
    """Raised when chart generation fails."""
    pass


class ExportError(RegressionAnalysisError):
    """Raised when export/output generation fails."""
    pass

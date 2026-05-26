"""Smoke tests verifying that core modules import correctly."""

import os

import pytest


class TestExceptions:
    def test_import(self):
        from src.utils.exceptions import (
            DataParseError,
            ExportError,
            ModelingError,
            RegressionAnalysisError,
            VisualizationError,
        )
        assert issubclass(DataParseError, RegressionAnalysisError)
        assert issubclass(ModelingError, RegressionAnalysisError)
        assert issubclass(VisualizationError, RegressionAnalysisError)
        assert issubclass(ExportError, RegressionAnalysisError)

    def test_raise(self):
        from src.utils.exceptions import DataParseError, RegressionAnalysisError
        with pytest.raises(RegressionAnalysisError):
            raise DataParseError("test error")


class TestLogger:
    def test_import(self):
        from src.utils.logger import get_logger
        logger = get_logger()
        assert logger is not None


class TestFixtures:
    def test_sample_df_shape(self, sample_df):
        assert sample_df.shape[0] == 100
        assert sample_df.shape[1] >= 7

    def test_sample_df_columns(self, sample_df):
        expected = {"y", "x1", "x2", "x3", "x4", "id", "cat1"}
        assert expected.issubset(set(sample_df.columns))

    def test_sample_df_has_missing(self, sample_df):
        assert sample_df["x4"].isna().sum() == 5

    def test_sample_csv_path_exists(self, sample_csv_path):
        assert os.path.exists(sample_csv_path)
        content = open(sample_csv_path, encoding="utf-8").read()
        assert "y,x1,x2,x3,x4,id,cat1" in content

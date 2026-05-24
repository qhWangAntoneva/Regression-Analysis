# encoding: utf-8
"""Tests for Phase 2 remaining features: file size limits, type override, data filtering."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.type_detector import VariableInfo, VariableTypeDetector


# =========================================================================
# File size limit constants
# =========================================================================

class TestFileSizeLimits:
    """Verify file size threshold constants (same values as 01_data_upload.py).

    Note: The actual constants live in app/pages/01_data_upload.py (numeric prefix),
    which cannot be imported as a Python module. These tests inline the same logic.
    """

    WARN_FILE_SIZE_MB = 50
    BLOCK_FILE_SIZE_MB = 200

    def _check_file_size(self, file_size_bytes: int) -> tuple[bool, str]:
        """Replica of _check_file_size from 01_data_upload.py."""
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > self.BLOCK_FILE_SIZE_MB:
            return False, f"文件过大（{size_mb:.1f} MB）。当前限制为最大 {self.BLOCK_FILE_SIZE_MB} MB。"
        if size_mb > self.WARN_FILE_SIZE_MB:
            return True, f"文件较大（{size_mb:.1f} MB），解析和处理可能需要较长时间。"
        return True, ""

    def test_warn_threshold_reasonable(self):
        assert self.WARN_FILE_SIZE_MB == 50

    def test_block_greater_than_warn(self):
        assert self.BLOCK_FILE_SIZE_MB > self.WARN_FILE_SIZE_MB

    def test_check_file_size_ok(self):
        is_ok, msg = self._check_file_size(1 * 1024 * 1024)  # 1 MB
        assert is_ok is True
        assert msg == ""

    def test_check_file_size_at_warn_boundary(self):
        """Exactly at warning threshold → no warning."""
        size = self.WARN_FILE_SIZE_MB * 1024 * 1024
        is_ok, msg = self._check_file_size(size)
        assert is_ok is True
        assert msg == ""

    def test_check_file_size_just_above_warn(self):
        size = (self.WARN_FILE_SIZE_MB + 1) * 1024 * 1024
        is_ok, msg = self._check_file_size(size)
        assert is_ok is True
        assert "文件较大" in msg

    def test_check_file_size_at_block_boundary(self):
        """Exactly at block threshold → not blocked (warning only)."""
        size = self.BLOCK_FILE_SIZE_MB * 1024 * 1024
        is_ok, msg = self._check_file_size(size)
        assert is_ok is True
        assert "文件较大" in msg

    def test_check_file_size_blocked(self):
        size = (self.BLOCK_FILE_SIZE_MB + 10) * 1024 * 1024
        is_ok, msg = self._check_file_size(size)
        assert is_ok is False
        assert "文件过大" in msg

    def test_edge_exact_zero(self):
        is_ok, msg = self._check_file_size(0)
        assert is_ok is True
        assert msg == ""


# =========================================================================
# Variable type override logic
# =========================================================================

class TestVariableTypeOverride:
    """Test that VariableInfo supports type changes."""

    def test_change_inferred_type(self, sample_df):
        """Verify VariableInfo.inferred_type can be changed."""
        detector = VariableTypeDetector()
        variables = detector.detect(sample_df)

        for v in variables:
            original = v.inferred_type
            # Change to a different valid type
            new_type = "categorical" if original != "categorical" else "continuous"
            v.inferred_type = new_type
            assert v.inferred_type == new_type

    def test_type_change_propagates_to_dict(self, sample_df):
        """Verify to_dict reflects updated type."""
        detector = VariableTypeDetector()
        variables = detector.detect(sample_df)

        var = variables[0]
        original = var.inferred_type
        new_type = "binary" if original != "binary" else "categorical"
        var.inferred_type = new_type

        d = var.to_dict()
        assert d["inferred_type"] == new_type

    def test_all_supported_types_can_be_assigned(self, sample_df):
        """All 6 types should be assignable to any variable."""
        detector = VariableTypeDetector()
        variables = detector.detect(sample_df)

        valid_types = ["continuous", "categorical", "binary", "ordinal", "id", "text"]
        for v in variables[:3]:
            for t in valid_types:
                v.inferred_type = t
                assert v.inferred_type == t


# =========================================================================
# Data filtering logic
# =========================================================================

class TestDataFiltering:
    """Test DataFrame filtering operations used in _render_data_filter."""

    def test_numeric_range_filter(self, sample_df):
        """Filter numeric column by min/max range."""
        x1_col = sample_df["x1"]
        x1_min, x1_max = float(x1_col.min()), float(x1_col.max())
        mid = (x1_min + x1_max) / 2

        filtered = sample_df[(sample_df["x1"] >= x1_min) & (sample_df["x1"] <= mid)]
        assert len(filtered) < len(sample_df)
        assert filtered["x1"].max() <= mid

    def test_categorical_filter(self, sample_df):
        """Filter categorical column by value selection."""
        filtered = sample_df[sample_df["x3"].isin(["A", "B"])]
        assert len(filtered) < len(sample_df)
        assert "C" not in filtered["x3"].unique()

    def test_binary_filter(self, sample_df):
        """Filter binary column by value."""
        filtered = sample_df[sample_df["cat1"] == 1]
        assert len(filtered) < len(sample_df)
        assert (filtered["cat1"] == 1).all()

    def test_composite_filter(self, sample_df):
        """Combine numeric and categorical filters."""
        step1 = sample_df[sample_df["x1"] > 0]
        step2 = step1[step1["x3"].isin(["A", "B"])]
        assert len(step2) <= len(step1)

    def test_filter_no_rows_matches(self, sample_df):
        """Filter that matches no rows returns empty DataFrame."""
        filtered = sample_df[sample_df["x1"] > 1e6]  # Impossible threshold
        assert len(filtered) == 0

    def test_filter_all_rows_match(self, sample_df):
        """Filter that matches all rows returns full DataFrame."""
        x1_max = sample_df["x1"].max()
        filtered = sample_df[sample_df["x1"] <= x1_max]
        assert len(filtered) == len(sample_df)

    def test_filter_columns_preserved(self, sample_df):
        """Filtering should preserve all columns and dtypes."""
        filtered = sample_df[sample_df["x1"] > 0]
        assert list(filtered.columns) == list(sample_df.columns)
        for col in sample_df.columns:
            assert filtered[col].dtype == sample_df[col].dtype

    def test_categorical_filter_empty_selection(self, sample_df):
        """Empty categorical selection returns no rows."""
        filtered = sample_df[sample_df["x3"].isin([])]
        assert len(filtered) == 0

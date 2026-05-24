# encoding: utf-8
"""Unit tests for the VariableTransformer.

Tests cover:
    - Log transform (including zero-safety)
    - Z-score standardize
    - Centering
    - Square term generation
    - Metadata structure
    - Validation errors
    - Interaction term addition
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.transforms import VariableTransformer


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def df() -> pd.DataFrame:
    """A small clean DataFrame for transform tests."""
    return pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x1": [10.0, 20.0, 30.0, 40.0, 50.0],
        "x2": [0.0, 1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "A", "B", "A"],
    })


# =========================================================================
# Test: Log transform
# =========================================================================

class TestLogTransform:
    def test_log_basic(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, meta = trans.transform(df, {"x1": "log"})
        # Should have a new column named "x1_log"
        assert "x1_log" in result.columns
        # log(10) ~= 2.3026
        assert abs(result["x1_log"].iloc[0] - np.log(10)) < 1e-6
        # Metadata check
        assert meta == {"x1": {"log": "x1_log"}}

    def test_log_zero_safe(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, _ = trans.transform(df, {"x2": "log"})
        # log(0) should be log(1e-10) not -inf
        assert np.isfinite(result["x2_log"].iloc[0])
        assert result["x2_log"].iloc[0] == pytest.approx(np.log(1e-10), abs=1e-6)

    def test_log_negative_safe(self) -> None:
        """log transform should handle negative values gracefully via max()."""
        df_neg = pd.DataFrame({"v": [-1.0, 0.0, 1.0]})
        trans = VariableTransformer()
        result, _ = trans.transform(df_neg, {"v": "log"})
        # -1 becomes 0 via max(col, 0), then log(1e-10)
        assert np.isfinite(result["v_log"].iloc[0])
        assert np.isfinite(result["v_log"].iloc[1])
        assert np.isfinite(result["v_log"].iloc[2])


# =========================================================================
# Test: Standardize (Z-score)
# =========================================================================

class TestStandardize:
    def test_standardize_mean_zero(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, meta = trans.transform(df, {"x1": "standardize"})
        col = result["x1_z"]
        assert abs(col.mean()) < 1e-10
        assert meta == {"x1": {"standardize": "x1_z"}}

    def test_standardize_std_one(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, _ = trans.transform(df, {"x1": "standardize"})
        col = result["x1_z"]
        assert abs(col.std(ddof=0) - 1.0) < 1e-10

    def test_standardize_constant(self) -> None:
        """Standardize of a constant column should not divide by zero."""
        df_const = pd.DataFrame({"v": [5.0, 5.0, 5.0]})
        trans = VariableTransformer()
        result, _ = trans.transform(df_const, {"v": "standardize"})
        # std is 0; fallback divides by 1, so all values become 0
        assert (result["v_z"] == 0.0).all()


# =========================================================================
# Test: Centering
# =========================================================================

class TestCenter:
    def test_center_mean_zero(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, meta = trans.transform(df, {"x1": "center"})
        col = result["x1_c"]
        assert abs(col.mean()) < 1e-10
        assert meta == {"x1": {"center": "x1_c"}}

    def test_center_preserves_std(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, _ = trans.transform(df, {"x1": "center"})
        # std should be the same as original
        orig_std = df["x1"].std(ddof=0)
        centered_std = result["x1_c"].std(ddof=0)
        assert abs(orig_std - centered_std) < 1e-10


# =========================================================================
# Test: Square
# =========================================================================

class TestSquare:
    def test_square_values(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, meta = trans.transform(df, {"x1": "square"})
        assert "x1_sq" in result.columns
        assert (result["x1_sq"] == df["x1"] ** 2).all()
        assert meta == {"x1": {"square": "x1_sq"}}


# =========================================================================
# Test: Multiple transforms
# =========================================================================

class TestMultipleTransforms:
    def test_two_vars_two_transforms(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        result, meta = trans.transform(df, {"x1": "log", "x2": "square"})
        assert "x1_log" in result.columns
        assert "x2_sq" in result.columns
        assert meta == {
            "x1": {"log": "x1_log"},
            "x2": {"square": "x2_sq"},
        }

    def test_original_data_preserved(self, df: pd.DataFrame) -> None:
        """Original columns should remain unchanged."""
        trans = VariableTransformer()
        result, _ = trans.transform(df, {"x1": "log"})
        assert (result["x1"] == df["x1"]).all()
        assert (result["y"] == df["y"]).all()


# =========================================================================
# Test: Validation
# =========================================================================

class TestValidation:
    def test_unknown_variable(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        with pytest.raises(ValueError, match="not found"):
            trans.transform(df, {"nonexistent": "log"})

    def test_unsupported_transform(self, df: pd.DataFrame) -> None:
        trans = VariableTransformer()
        with pytest.raises(ValueError, match="Unsupported"):
            trans.transform(df, {"x1": "sqrt"})


# =========================================================================
# Test: Interaction terms
# =========================================================================

class TestInteractionTerms:
    def test_add_interaction(self, df: pd.DataFrame) -> None:
        result, names = VariableTransformer.add_interactions(
            df, [("x1", "x2")]
        )
        assert names == ["x1_x_x2"]
        assert "x1_x_x2" in result.columns
        # Check product values
        expected = df["x1"] * df["x2"]
        assert (result["x1_x_x2"] == expected).all()

    def test_multiple_interactions(self, df: pd.DataFrame) -> None:
        result, names = VariableTransformer.add_interactions(
            df, [("x1", "x2"), ("x1", "y")]
        )
        assert len(names) == 2
        assert "x1_x_x2" in result.columns
        assert "x1_x_y" in result.columns

    def test_interaction_missing_variable(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            VariableTransformer.add_interactions(
                df, [("x1", "nonexistent")]
            )

    def test_original_data_unchanged(self, df: pd.DataFrame) -> None:
        result, _ = VariableTransformer.add_interactions(df, [("x1", "x2")])
        assert (result["x1"] == df["x1"]).all()
        assert (result["x2"] == df["x2"]).all()

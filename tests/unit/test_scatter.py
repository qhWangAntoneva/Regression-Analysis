"""Tests for scatter_with_regression visualization in src/visualization/scatter.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.visualization.scatter import scatter_with_regression


@pytest.fixture
def simple_df():
    """A simple 10-row DataFrame with numeric x, y and a group column."""
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [2.1, 4.0, 5.8, 8.2, 10.1, 12.0, 14.2, 15.9, 18.1, 20.3],
        "group": ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C"],
    })


class TestBasicPlot:
    def test_returns_figure(self, simple_df):
        fig = scatter_with_regression(simple_df, x_col="x", y_col="y")
        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_has_expected_traces(self, simple_df):
        fig = scatter_with_regression(simple_df, x_col="x", y_col="y")
        # At least scatter + trendline traces
        assert len(fig.data) >= 2

    def test_auto_title(self, simple_df):
        fig = scatter_with_regression(simple_df, x_col="x", y_col="y")
        assert "y vs x" == fig.layout.title.text

    def test_custom_title(self, simple_df):
        fig = scatter_with_regression(
            simple_df, x_col="x", y_col="y", title="Custom Title"
        )
        assert fig.layout.title.text == "Custom Title"

    def test_figure_has_plotly_white_template(self, simple_df):
        fig = scatter_with_regression(simple_df, x_col="x", y_col="y")
        assert fig.layout.template is not None


class TestColorGrouping:
    def test_color_column_adds_group_traces(self, simple_df):
        fig = scatter_with_regression(
            simple_df, x_col="x", y_col="y", color_col="group"
        )
        assert len(fig.data) >= 2

    def test_legend_title_set(self, simple_df):
        fig = scatter_with_regression(
            simple_df, x_col="x", y_col="y", color_col="group"
        )
        assert fig.layout.legend.title.text == "group"


class TestErrorHandling:
    def test_missing_x_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="数据中不存在列"):
            scatter_with_regression(simple_df, x_col="nonexistent", y_col="y")

    def test_missing_y_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="数据中不存在列"):
            scatter_with_regression(simple_df, x_col="x", y_col="nonexistent")

    def test_missing_color_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="数据中不存在列"):
            scatter_with_regression(
                simple_df, x_col="x", y_col="y", color_col="nonexistent"
            )

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": [], "y": []})
        with pytest.raises(ValueError, match="数据点不足"):
            scatter_with_regression(df, x_col="x", y_col="y")

    def test_single_data_point_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="数据点不足"):
            scatter_with_regression(df, x_col="x", y_col="y")

    def test_single_missing_value_column(self):
        """DataFrame with 2 rows but 1 NaN -> effectively 1 valid point."""
        df = pd.DataFrame({"x": [1, None], "y": [2, 4]})
        with pytest.raises(ValueError, match="数据点不足"):
            scatter_with_regression(df, x_col="x", y_col="y")


class TestDataWithNaNs:
    def test_drop_na_rows(self):
        df = pd.DataFrame({
            "x": [1, 2, None, 4, 5],
            "y": [2, 4, 6, None, 10],
        })
        fig = scatter_with_regression(df, x_col="x", y_col="y")
        assert len(fig.data) >= 2

    def test_color_with_nans(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "group": ["A", None, "B", "B", "A"],
        })
        fig = scatter_with_regression(df, x_col="x", y_col="y", color_col="group")
        assert len(fig.data) >= 2

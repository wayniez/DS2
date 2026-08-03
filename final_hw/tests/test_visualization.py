"""Tests for `app.tools.visualization`.

Requires the `plotly` package. Not runnable in this sandbox (no network
access to install it), but structured to run cleanly once dependencies
are installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import InvalidToolArgumentsError
from app.tools.visualization import create_visualization


class TestCreateVisualization:
    def test_histogram(self, churn_df: pd.DataFrame) -> None:
        result = create_visualization(churn_df, "histogram", column="tenure")
        assert result["chart_type"] == "histogram"
        assert "data" in result["plotly_spec"]

    def test_box_plot_grouped(self, churn_df: pd.DataFrame) -> None:
        result = create_visualization(churn_df, "box", column="monthly_charges", group_by="contract")
        assert result["chart_type"] == "box"

    def test_scatter_plot(self, churn_df: pd.DataFrame) -> None:
        result = create_visualization(churn_df, "scatter", x="tenure", y="monthly_charges")
        assert result["chart_type"] == "scatter"

    def test_bar_chart_counts(self, churn_df: pd.DataFrame) -> None:
        result = create_visualization(churn_df, "bar", category_column="contract")
        assert result["chart_type"] == "bar"

    def test_correlation_heatmap(self, churn_df: pd.DataFrame) -> None:
        result = create_visualization(churn_df, "correlation_heatmap")
        assert result["chart_type"] == "correlation_heatmap"

    def test_category_comparison(self, churn_df: pd.DataFrame) -> None:
        result = create_visualization(
            churn_df, "category_comparison", category_column="contract", target_column="churn"
        )
        assert result["chart_type"] == "category_comparison"

    def test_unsupported_chart_type_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            create_visualization(churn_df, "pie_of_doom", column="tenure")

    def test_unknown_column_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            create_visualization(churn_df, "histogram", column="does_not_exist")

    def test_correlation_heatmap_insufficient_columns_raises(self) -> None:
        df = pd.DataFrame({"only_col": [1, 2, 3]})
        with pytest.raises(InvalidToolArgumentsError):
            create_visualization(df, "correlation_heatmap")

    def test_each_chart_has_unique_id(self, churn_df: pd.DataFrame) -> None:
        r1 = create_visualization(churn_df, "histogram", column="tenure")
        r2 = create_visualization(churn_df, "histogram", column="tenure")
        assert r1["chart_id"] != r2["chart_id"]

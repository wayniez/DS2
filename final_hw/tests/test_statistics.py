"""Tests for `app.tools.statistics`."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import InvalidToolArgumentsError
from app.tools.statistics import correlation_matrix, describe_column, group_statistics


class TestDescribeColumn:
    def test_numeric_column_summary(self, churn_df: pd.DataFrame) -> None:
        result = describe_column(churn_df, "tenure")
        assert result["type"] == "numeric"
        assert "mean" in result and "median" in result and "std" in result

    def test_categorical_column_summary(self, churn_df: pd.DataFrame) -> None:
        result = describe_column(churn_df, "contract")
        assert result["type"] == "categorical"
        assert "distribution" in result

    def test_missing_values_reported(self, churn_df: pd.DataFrame) -> None:
        result = describe_column(churn_df, "monthly_charges")
        assert result["missing"] == 3

    def test_unknown_column_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            describe_column(churn_df, "does_not_exist")


class TestGroupStatistics:
    def test_numeric_target_group_means(self, churn_df: pd.DataFrame) -> None:
        result = group_statistics(churn_df, "contract", "monthly_charges", "mean")
        assert result["target_type"] == "numeric"
        assert set(result["results"].keys()) == {"Month-to-month", "One year", "Two year"}

    def test_categorical_target_rates(self, churn_df: pd.DataFrame) -> None:
        result = group_statistics(churn_df, "contract", "churn")
        assert result["target_type"] == "categorical"
        # Every group's category rates should sum to ~100%.
        for group, rates in result["results"].items():
            assert abs(sum(rates.values()) - 100.0) < 0.5

    def test_month_to_month_has_higher_churn_rate(self, churn_df: pd.DataFrame) -> None:
        # Regression/signal guard: the synthetic fixture is constructed so
        # that month-to-month contracts churn more than one-year contracts.
        result = group_statistics(churn_df, "contract", "churn")
        mtm_rate = result["results"]["Month-to-month"].get("Yes", 0)
        one_year_rate = result["results"]["One year"].get("Yes", 0)
        assert mtm_rate > one_year_rate

    def test_invalid_group_column_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            group_statistics(churn_df, "does_not_exist", "churn")

    def test_invalid_aggregation_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            group_statistics(churn_df, "contract", "monthly_charges", agg="not_a_real_agg")


class TestCorrelationMatrix:
    def test_default_numeric_columns(self, churn_df: pd.DataFrame) -> None:
        result = correlation_matrix(churn_df)
        assert "tenure" in result["columns"]
        assert "monthly_charges" in result["columns"]

    def test_insufficient_numeric_columns_returns_error(self, churn_df: pd.DataFrame) -> None:
        result = correlation_matrix(churn_df, columns=["contract"])
        assert "error" in result

    def test_unknown_column_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            correlation_matrix(churn_df, columns=["does_not_exist"])

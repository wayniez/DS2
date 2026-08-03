"""Tests for `app.analytics.validation` and `app.analytics.profiling`.

These exercise pure pandas/numpy logic with no LLM, FastAPI, or DuckDB
dependency, so they can run in any environment with the base
scientific-Python stack installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.analytics.profiling import detect_possible_targets, profile_columns, profile_dataset
from app.analytics.validation import validate_dataframe, validate_target_column
from app.core.exceptions import DatasetTooLargeError, EmptyDatasetError, InvalidCSVError


class TestValidation:
    def test_valid_dataframe_passes(self, churn_df: pd.DataFrame) -> None:
        # Should not raise.
        validate_dataframe(churn_df, max_rows=10_000)

    def test_empty_dataframe_raises(self, empty_df: pd.DataFrame) -> None:
        with pytest.raises((EmptyDatasetError, InvalidCSVError)):
            validate_dataframe(empty_df, max_rows=10_000)

    def test_zero_row_dataframe_raises(self) -> None:
        df = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(EmptyDatasetError):
            validate_dataframe(df, max_rows=10_000)

    def test_oversized_dataframe_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(DatasetTooLargeError):
            validate_dataframe(churn_df, max_rows=5)

    def test_missing_target_column_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidCSVError):
            validate_target_column(churn_df, "does_not_exist")

    def test_existing_target_column_passes(self, churn_df: pd.DataFrame) -> None:
        validate_target_column(churn_df, "churn")


class TestProfiling:
    def test_profile_dataset_row_column_counts(self, churn_df: pd.DataFrame) -> None:
        result = profile_dataset(churn_df)
        assert result["n_rows"] == len(churn_df)
        assert result["n_columns"] == churn_df.shape[1]

    def test_duplicate_rows_detected(self, churn_df: pd.DataFrame) -> None:
        result = profile_dataset(churn_df)
        assert result["duplicate_row_count"] >= 1

    def test_numeric_columns_detected(self, churn_df: pd.DataFrame) -> None:
        result = profile_dataset(churn_df)
        assert "tenure" in result["numeric_columns"]
        assert "monthly_charges" in result["numeric_columns"]

    def test_categorical_columns_detected(self, churn_df: pd.DataFrame) -> None:
        result = profile_dataset(churn_df)
        assert "contract" in result["categorical_columns"]
        assert "churn" in result["categorical_columns"]

    def test_datetime_column_detected(self, churn_df: pd.DataFrame) -> None:
        result = profile_dataset(churn_df)
        assert "signup_date" in result["datetime_columns"]

    def test_high_cardinality_id_not_categorical(self, churn_df: pd.DataFrame) -> None:
        result = profile_dataset(churn_df)
        assert "customer_id" not in result["categorical_columns"]
        assert "customer_id" not in result["numeric_columns"]

    def test_missing_values_counted(self, churn_df: pd.DataFrame) -> None:
        profiles = profile_columns(churn_df)
        monthly = next(p for p in profiles if p["name"] == "monthly_charges")
        assert monthly["missing_count"] == 3

    def test_numeric_summary_present_for_numeric_columns(self, churn_df: pd.DataFrame) -> None:
        profiles = profile_columns(churn_df)
        tenure = next(p for p in profiles if p["name"] == "tenure")
        assert tenure["numeric_summary"] is not None
        assert "mean" in tenure["numeric_summary"]
        assert "median" in tenure["numeric_summary"]

    def test_top_categories_present_for_categorical_columns(self, churn_df: pd.DataFrame) -> None:
        profiles = profile_columns(churn_df)
        contract = next(p for p in profiles if p["name"] == "contract")
        assert contract["top_categories"] is not None
        assert sum(contract["top_categories"].values()) <= len(churn_df)

    def test_possible_targets_includes_churn(self, churn_df: pd.DataFrame) -> None:
        profiles = profile_columns(churn_df)
        candidates = detect_possible_targets(profiles)
        assert "churn" in candidates

    def test_possible_targets_excludes_continuous_numeric_with_y_substring(
        self, churn_df: pd.DataFrame
    ) -> None:
        # Regression guard: a column like "monthly_charges" must never be
        # suggested as a target just because it contains the letter "y".
        profiles = profile_columns(churn_df)
        candidates = detect_possible_targets(profiles)
        assert "monthly_charges" not in candidates

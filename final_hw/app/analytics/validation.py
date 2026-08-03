"""Validation helpers for uploaded datasets.

These checks run before any profiling/analysis and are responsible for
turning "bad input" into clear, typed errors (Section 13) instead of
opaque pandas exceptions bubbling up later in the pipeline.
"""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import DatasetTooLargeError, EmptyDatasetError, InvalidCSVError


def validate_dataframe(df: pd.DataFrame, max_rows: int) -> None:
    """Run structural sanity checks on a freshly-loaded dataframe.

    Raises:
        EmptyDatasetError: if there are no rows or no columns.
        DatasetTooLargeError: if the dataset exceeds `max_rows`.
    """
    if df.shape[1] == 0:
        raise InvalidCSVError("The uploaded file has no columns; it may not be a valid CSV.")

    if df.shape[0] == 0:
        raise EmptyDatasetError("The uploaded dataset has no rows.")

    if df.shape[0] > max_rows:
        raise DatasetTooLargeError(
            f"Dataset has {df.shape[0]} rows, which exceeds the configured "
            f"limit of {max_rows}. Please upload a smaller sample."
        )


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Validate that a column chosen as an ML target is usable.

    Raises:
        InvalidCSVError: if the column does not exist.
    """
    if target_column not in df.columns:
        raise InvalidCSVError(f"Target column '{target_column}' does not exist in the dataset.")

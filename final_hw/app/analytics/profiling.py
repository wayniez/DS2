"""Dataset profiling logic.

This module contains the actual computation behind `inspect_dataset()`
and `profile_dataset()`. It intentionally has no dependency on pydantic
or the LLM layer -- it operates purely on pandas/numpy and returns plain
Python dicts/lists, which the tool layer (app/tools/dataset.py) then
wraps into the `DatasetMetadata` schema.

Keeping this separation means the profiling logic can be exercised in
tests without needing the rest of the stack installed.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# Column name substrings that are common indicators of a target/label
# column. This is a lightweight heuristic used to *suggest* candidates
# to the LLM -- it never forces a particular target.
_TARGET_NAME_HINTS = (
    "churn",
    "target",
    "label",
    "class",
    "outcome",
    "default",
    "fraud",
    "converted",
)

# A column with this few unique values (regardless of dtype) is treated
# as categorical rather than numeric/continuous, even if it's stored as
# an int (e.g. a 0/1 flag column).
_MAX_CATEGORICAL_UNIQUE = 20
_MAX_CATEGORICAL_RATIO = 0.05


def _detect_column_type(series: pd.Series, n_rows: int) -> str:
    """Classify a column into a coarse semantic type.

    Order of checks matters: boolean/datetime are checked before
    numeric/categorical since pandas may represent them with numeric-like
    dtypes.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # A column is "string-like" if it's the legacy object dtype, pandas'
    # newer dedicated string dtype (default in pandas >= 2.x/3.x for CSV
    # text columns), or an explicit categorical dtype. Checking dtype
    # equality against `object` alone misses the modern string dtype.
    is_string_like = (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    # Attempt lightweight datetime detection for string-like columns that
    # weren't already parsed as dates (e.g. "2023-01-05" strings).
    if is_string_like:
        non_null = series.dropna()
        if len(non_null) > 0:
            sample = non_null.head(min(50, len(non_null)))
            try:
                parsed = pd.to_datetime(sample, errors="raise", format="mixed")
                if parsed.notna().all():
                    return "datetime"
            except (ValueError, TypeError):
                pass

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        ratio = n_unique / n_rows if n_rows else 0
        if n_unique <= _MAX_CATEGORICAL_UNIQUE and ratio <= _MAX_CATEGORICAL_RATIO:
            return "categorical"
        return "numeric"

    if is_string_like:
        n_unique = series.nunique(dropna=True)
        # High-cardinality string columns (free text, IDs) are marked as
        # "text" rather than "categorical" so the agent doesn't try to
        # group-by or one-hot-encode an ID column.
        if n_unique > 0 and n_unique / n_rows > 0.5 and n_unique > 50:
            return "text"
        return "categorical"

    return "unknown"


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {}
    desc = clean.describe()
    summary = {
        "mean": float(desc.get("mean", np.nan)),
        "std": float(desc.get("std", np.nan)),
        "min": float(desc.get("min", np.nan)),
        "p25": float(desc.get("25%", np.nan)),
        "median": float(desc.get("50%", np.nan)),
        "p75": float(desc.get("75%", np.nan)),
        "max": float(desc.get("max", np.nan)),
    }
    return {k: (round(v, 4) if v == v else 0.0) for k, v in summary.items()}  # NaN check


def _top_categories(series: pd.Series, top_n: int = 10) -> dict[str, int]:
    clean = series.dropna()
    counts = clean.astype(str).value_counts().head(top_n)
    return {str(k): int(v) for k, v in counts.items()}


def _sample_values(series: pd.Series, n: int = 3) -> list[str]:
    clean = series.dropna()
    if clean.empty:
        return []
    sample = clean.head(n)
    return [str(v) for v in sample.tolist()]


def profile_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Produce a per-column profile for every column in the dataframe."""
    n_rows = len(df)
    profiles: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        col_type = _detect_column_type(series, n_rows)
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        profile: dict[str, Any] = {
            "name": str(col),
            "pandas_dtype": str(series.dtype),
            "column_type": col_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_count / n_rows * 100, 2) if n_rows else 0.0,
            "unique_count": unique_count,
            "cardinality_ratio": round(unique_count / n_rows, 4) if n_rows else 0.0,
            "sample_values": _sample_values(series),
            "numeric_summary": None,
            "top_categories": None,
        }

        if col_type == "numeric":
            profile["numeric_summary"] = _numeric_summary(series)
        elif col_type in ("categorical", "boolean"):
            profile["top_categories"] = _top_categories(series)

        profiles.append(profile)

    return profiles


def detect_possible_targets(column_profiles: list[dict[str, Any]]) -> list[str]:
    """Heuristically suggest candidate target/label columns.

    A column is suggested if either its name matches a common target
    naming pattern, or it is a low-cardinality (binary/few-class)
    categorical/boolean column -- both are typical shapes for a
    classification target in this kind of dataset.
    """
    candidates: list[str] = []
    for profile in column_profiles:
        name_tokens = set(re.split(r"[^a-z0-9]+", profile["name"].lower()))
        name_hint = any(hint in name_tokens for hint in _TARGET_NAME_HINTS)
        low_cardinality_categorical = profile["column_type"] in ("categorical", "boolean") and profile[
            "unique_count"
        ] <= 10

        if name_hint or low_cardinality_categorical:
            candidates.append(profile["name"])

    return candidates


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Produce the full dataset-level profile used to build `DatasetMetadata`."""
    column_profiles = profile_columns(df)

    numeric_columns = [p["name"] for p in column_profiles if p["column_type"] == "numeric"]
    categorical_columns = [p["name"] for p in column_profiles if p["column_type"] == "categorical"]
    datetime_columns = [p["name"] for p in column_profiles if p["column_type"] == "datetime"]
    boolean_columns = [p["name"] for p in column_profiles if p["column_type"] == "boolean"]

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": column_profiles,
        "duplicate_row_count": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
        "possible_target_columns": detect_possible_targets(column_profiles),
    }

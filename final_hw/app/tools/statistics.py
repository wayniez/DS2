"""Statistics tool (Section 6, tool #3: `calculate_statistics`).

Pure pandas computation. This module has no LLM or pydantic dependency
so it can be tested in isolation. The tool_registry wraps these
functions with a JSON schema for LLM tool-calling.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.exceptions import InvalidToolArgumentsError

_VALID_STATS = ("mean", "median", "std", "min", "max", "count", "sum")


def describe_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return mean/median/std/quantiles/count for a single numeric column,
    or value counts / distribution for a categorical column.
    """
    if column not in df.columns:
        raise InvalidToolArgumentsError(f"Column '{column}' not found in dataset.")

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if clean.empty:
            return {"column": column, "type": "numeric", "error": "All values are missing."}
        return {
            "column": column,
            "type": "numeric",
            "count": int(clean.count()),
            "missing": int(series.isna().sum()),
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "q25": round(float(clean.quantile(0.25)), 4),
            "q75": round(float(clean.quantile(0.75)), 4),
        }

    clean = series.dropna().astype(str)
    value_counts = clean.value_counts()
    return {
        "column": column,
        "type": "categorical",
        "count": int(clean.count()),
        "missing": int(series.isna().sum()),
        "n_unique": int(value_counts.shape[0]),
        "distribution": {str(k): int(v) for k, v in value_counts.head(20).items()},
        "distribution_pct": {
            str(k): round(float(v) / len(clean) * 100, 2) for k, v in value_counts.head(20).items()
        },
    }


def group_statistics(
    df: pd.DataFrame,
    group_by: str,
    target: str,
    agg: str = "mean",
) -> dict[str, Any]:
    """Compute an aggregate statistic of `target`, grouped by `group_by`.

    This is the core operation behind questions like "which contract type
    has the highest churn rate" or "average monthly charges per segment".
    For a categorical `target`, `agg` is ignored and this instead returns
    the rate of each target category within each group (e.g. churn rate).
    """
    if group_by not in df.columns:
        raise InvalidToolArgumentsError(f"Group-by column '{group_by}' not found in dataset.")
    if target not in df.columns:
        raise InvalidToolArgumentsError(f"Target column '{target}' not found in dataset.")
    if agg not in _VALID_STATS:
        raise InvalidToolArgumentsError(f"Unsupported aggregation '{agg}'. Valid: {_VALID_STATS}")

    working = df[[group_by, target]].dropna()

    if pd.api.types.is_numeric_dtype(working[target]):
        grouped = working.groupby(group_by, observed=True)[target].agg(agg)
        return {
            "group_by": group_by,
            "target": target,
            "aggregation": agg,
            "target_type": "numeric",
            "results": {str(k): round(float(v), 4) for k, v in grouped.items()},
            "group_sizes": {
                str(k): int(v) for k, v in working.groupby(group_by, observed=True).size().items()
            },
        }

    # Categorical target -> report the rate of each category per group.
    # This directly answers "which group has the highest churn rate".
    crosstab = pd.crosstab(working[group_by], working[target], normalize="index") * 100
    counts = pd.crosstab(working[group_by], working[target])
    return {
        "group_by": group_by,
        "target": target,
        "aggregation": "rate_pct",
        "target_type": "categorical",
        "results": {str(idx): {str(c): round(float(v), 2) for c, v in row.items()} for idx, row in crosstab.iterrows()},
        "counts": {str(idx): {str(c): int(v) for c, v in row.items()} for idx, row in counts.iterrows()},
        "group_sizes": {str(k): int(v) for k, v in working.groupby(group_by, observed=True).size().items()},
    }


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """Compute a Pearson correlation matrix over numeric columns.

    If `columns` is omitted, all numeric columns are used.
    """
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise InvalidToolArgumentsError(f"Columns not found: {missing}")
        numeric_df = df[columns].select_dtypes(include="number")
    else:
        numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return {
            "error": "Need at least two numeric columns to compute a correlation matrix.",
            "numeric_columns_available": list(numeric_df.columns),
        }

    corr = numeric_df.corr(numeric_only=True).round(4)
    return {
        "columns": list(corr.columns),
        "matrix": {col: corr[col].to_dict() for col in corr.columns},
    }

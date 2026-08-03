"""SQL analytics tool (Section 6, tool #4: `run_sql`).

Uses an in-memory DuckDB connection scoped to a single session/dataset.
The dataset is registered as a DuckDB view named `dataset`, and every
query is validated before execution so that the LLM can only run
read-only SELECT queries against that single view -- never arbitrary
filesystem, catalog, or multi-database access (Section 6, 14).
"""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from app.core.exceptions import SQLValidationError

# The only table name the LLM is allowed to reference. Enforced by both
# instructing the LLM (via the tool schema/prompt) and validating the
# query text before execution.
DATASET_VIEW_NAME = "dataset"

# Maximum rows returned to the LLM/frontend from a single query, to keep
# context small (Section 5/8) and avoid huge payloads.
MAX_RESULT_ROWS = 500

# Keywords that indicate a write/DDL/attach statement rather than a
# read-only SELECT. Blocking these (in addition to only ever opening an
# in-memory, single-table connection) is defense in depth.
_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "attach",
    "detach",
    "copy",
    "export",
    "import",
    "pragma",
    "install",
    "load",
    "call",
    "vacuum",
    "grant",
    "revoke",
)


def _validate_query(query: str) -> None:
    """Reject anything that isn't a single, plain SELECT statement."""
    stripped = query.strip().rstrip(";")

    if not stripped:
        raise SQLValidationError("Empty query.")

    # Only one statement allowed (no chained statements via ';').
    if ";" in stripped:
        raise SQLValidationError("Multiple statements are not allowed; submit a single SELECT query.")

    if not re.match(r"^\s*(with\b|select\b)", stripped, flags=re.IGNORECASE):
        raise SQLValidationError("Only SELECT (optionally with a leading WITH/CTE) queries are allowed.")

    lowered = stripped.lower()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise SQLValidationError(
                f"Query contains forbidden keyword '{keyword}'. Only read-only SELECT queries "
                "against the current dataset are permitted."
            )


def run_sql(df: pd.DataFrame, query: str) -> dict[str, Any]:
    """Execute a validated, read-only SQL query against `df`.

    The dataframe is registered as a DuckDB view called `dataset` inside
    a fresh, in-memory, session-scoped connection -- the query has no
    access to the filesystem, other databases, or any table besides this
    one view.
    """
    _validate_query(query)

    con = duckdb.connect(database=":memory:")
    try:
        con.register(DATASET_VIEW_NAME, df)
        try:
            result_df = con.execute(query).fetchdf()
        except duckdb.Error as exc:
            raise SQLValidationError(f"SQL execution failed: {exc}") from exc
    finally:
        con.close()

    truncated = len(result_df) > MAX_RESULT_ROWS
    if truncated:
        result_df = result_df.head(MAX_RESULT_ROWS)

    return {
        "query": query,
        "columns": list(result_df.columns),
        "rows": result_df.to_dict(orient="records"),
        "row_count": int(len(result_df)),
        "truncated": truncated,
    }

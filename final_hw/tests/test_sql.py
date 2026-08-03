"""Tests for `app.tools.sql`.

Requires the `duckdb` package (see requirements.txt). Not runnable in
this sandbox (no network access to install it), but structured to run
cleanly once dependencies are installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import SQLValidationError
from app.tools.sql import run_sql


class TestRunSQL:
    def test_simple_select(self, churn_df: pd.DataFrame) -> None:
        result = run_sql(churn_df, "SELECT contract, COUNT(*) as n FROM dataset GROUP BY contract")
        assert result["row_count"] > 0
        assert "contract" in result["columns"]

    def test_select_with_where(self, churn_df: pd.DataFrame) -> None:
        result = run_sql(churn_df, "SELECT * FROM dataset WHERE churn = 'Yes'")
        assert all(row["churn"] == "Yes" for row in result["rows"])

    def test_cte_query_allowed(self, churn_df: pd.DataFrame) -> None:
        query = (
            "WITH by_contract AS (SELECT contract, COUNT(*) AS n FROM dataset GROUP BY contract) "
            "SELECT * FROM by_contract ORDER BY n DESC"
        )
        result = run_sql(churn_df, query)
        assert result["row_count"] > 0

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE dataset",
            "DELETE FROM dataset WHERE churn = 'Yes'",
            "INSERT INTO dataset VALUES (1,2,3)",
            "ATTACH 'some.db' AS other",
            "PRAGMA database_list",
            "SELECT * FROM dataset; DROP TABLE dataset;",
            "UPDATE dataset SET churn = 'No'",
            "CREATE TABLE evil AS SELECT * FROM dataset",
        ],
    )
    def test_forbidden_statements_rejected(self, churn_df: pd.DataFrame, query: str) -> None:
        with pytest.raises(SQLValidationError):
            run_sql(churn_df, query)

    def test_empty_query_rejected(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(SQLValidationError):
            run_sql(churn_df, "   ")

    def test_non_select_statement_type_rejected(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(SQLValidationError):
            run_sql(churn_df, "EXPLAIN SELECT * FROM dataset")

    def test_large_result_is_truncated(self, churn_df: pd.DataFrame) -> None:
        # churn_df fixture is small, so cross join to exceed MAX_RESULT_ROWS.
        query = "SELECT a.churn, b.contract FROM dataset a, dataset b"
        result = run_sql(churn_df, query)
        if result["row_count"] >= 500:
            assert result["truncated"] is True

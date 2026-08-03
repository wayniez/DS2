"""Shared pytest fixtures.

Fixtures here build small, deterministic in-memory dataframes so tests
do not depend on the sample CSV file on disk (that file is exercised
separately by integration-style tests / the evaluation harness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def churn_df() -> pd.DataFrame:
    """A small, deterministic churn-like dataframe with real signal.

    Contains: a numeric column, several categorical columns, a datetime
    column, a high-cardinality id column, missing values, and duplicate
    rows -- enough to exercise the full profiling/validation logic.
    """
    rng = np.random.default_rng(0)
    n = 120

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.6, 0.25, 0.15])
    tenure = rng.integers(0, 72, size=n)
    monthly_charges = np.round(rng.normal(70, 20, size=n).clip(15, 150), 2)
    churn_logit = -1.2 + np.where(contract == "Month-to-month", 1.5, 0.0) - 0.03 * tenure
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "contract": contract,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "signup_date": pd.date_range("2022-01-01", periods=n, freq="3D").astype(str),
            "churn": churn,
        }
    )

    # a few missing values
    df.loc[[1, 5, 9], "monthly_charges"] = np.nan
    # a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    return df


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def single_class_df() -> pd.DataFrame:
    """A dataframe whose target column has only one class -- used to test
    the "insufficient classes for classification" error path.
    """
    return pd.DataFrame(
        {
            "feature_a": range(20),
            "feature_b": np.random.default_rng(1).normal(size=20),
            "target": ["Yes"] * 20,
        }
    )

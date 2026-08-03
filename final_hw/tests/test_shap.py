"""Tests for `app.tools.shap_analysis`.

Requires the `shap` package. Not runnable in this sandbox (no network
access to install it), but structured to run cleanly once dependencies
are installed.
"""

from __future__ import annotations

import pandas as pd

from app.tools.ml import train_baseline_model
from app.tools.shap_analysis import calculate_shap


class TestCalculateShap:
    def test_shap_returns_top_features(self, churn_df: pd.DataFrame) -> None:
        train_result = train_baseline_model(churn_df, target_column="churn", use_xgboost=False)
        result = calculate_shap(train_result)
        assert len(result["top_features"]) > 0
        for feature in result["top_features"]:
            assert "mean_abs_shap" in feature
            assert "direction" in feature

    def test_shap_features_sorted_by_importance(self, churn_df: pd.DataFrame) -> None:
        train_result = train_baseline_model(churn_df, target_column="churn", use_xgboost=False)
        result = calculate_shap(train_result)
        values = [f["mean_abs_shap"] for f in result["top_features"]]
        assert values == sorted(values, reverse=True)

    def test_shap_on_regression_model(self, churn_df: pd.DataFrame) -> None:
        train_result = train_baseline_model(churn_df, target_column="monthly_charges", use_xgboost=False)
        result = calculate_shap(train_result)
        assert result["model"] == train_result["best_model"]

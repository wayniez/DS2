"""Tests for `app.tools.ml`."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import (
    InsufficientDataError,
    InvalidToolArgumentsError,
    ModelTrainingError,
)
from app.tools.ml import calculate_feature_importance, detect_anomalies, train_baseline_model


class TestTrainBaselineModel:
    def test_classification_auto_detected(self, churn_df: pd.DataFrame) -> None:
        result = train_baseline_model(churn_df, target_column="churn", use_xgboost=False)
        assert result["problem_type"] == "classification"
        assert "logistic_regression" in result["models"]
        assert "random_forest" in result["models"]

    def test_classification_metrics_present(self, churn_df: pd.DataFrame) -> None:
        result = train_baseline_model(churn_df, target_column="churn", use_xgboost=False)
        best = result["models"][result["best_model"]]
        assert "accuracy" in best
        assert "f1" in best

    def test_regression_auto_detected(self, churn_df: pd.DataFrame) -> None:
        result = train_baseline_model(churn_df, target_column="monthly_charges", use_xgboost=False)
        assert result["problem_type"] == "regression"
        assert "linear_regression" in result["models"]

    def test_regression_metrics_present(self, churn_df: pd.DataFrame) -> None:
        result = train_baseline_model(churn_df, target_column="monthly_charges", use_xgboost=False)
        best = result["models"][result["best_model"]]
        assert "r2" in best
        assert "mae" in best

    def test_unknown_target_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidToolArgumentsError):
            train_baseline_model(churn_df, target_column="does_not_exist")

    def test_single_class_target_raises(self, single_class_df: pd.DataFrame) -> None:
        with pytest.raises(ModelTrainingError):
            train_baseline_model(single_class_df, target_column="target")

    def test_too_few_rows_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            train_baseline_model(churn_df.head(10), target_column="churn")


class TestFeatureImportance:
    def test_returns_ranked_features(self, churn_df: pd.DataFrame) -> None:
        train_result = train_baseline_model(churn_df, target_column="churn", use_xgboost=False)
        fi = calculate_feature_importance(train_result)
        assert len(fi["top_features"]) > 0
        importances = [f["importance"] for f in fi["top_features"]]
        assert importances == sorted(importances, reverse=True)

    def test_contract_is_a_top_feature(self, churn_df: pd.DataFrame) -> None:
        # Regression/signal guard: the fixture is designed so contract type
        # is a strong predictor of churn.
        train_result = train_baseline_model(churn_df, target_column="churn", use_xgboost=False)
        fi = calculate_feature_importance(train_result, top_n=5)
        feature_names = [f["feature"] for f in fi["top_features"]]
        assert any("contract" in name.lower() for name in feature_names)


class TestDetectAnomalies:
    def test_returns_anomalies(self, churn_df: pd.DataFrame) -> None:
        result = detect_anomalies(churn_df)
        assert result["n_anomalies"] >= 0
        assert 0 <= result["anomaly_pct"] <= 100

    def test_no_numeric_columns_raises(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "z"] * 15})
        with pytest.raises(InvalidToolArgumentsError):
            detect_anomalies(df)

    def test_too_few_rows_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError):
            detect_anomalies(churn_df.head(5))

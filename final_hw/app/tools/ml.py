"""ML tools (Section 6, tools #7/#8/#10).

Covers:
- `train_baseline_model`: auto-detects classification vs regression and
  trains a small set of standard baseline models.
- `calculate_feature_importance`: model-based feature importance for the
  best baseline model.
- `detect_anomalies`: Isolation Forest based anomaly detection.

XGBoost is used opportunistically (Section 2 says "if appropriate"): if
it isn't installed, the module still works with sklearn models only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.core.exceptions import InsufficientDataError, InvalidToolArgumentsError, ModelTrainingError
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _XGBOOST_AVAILABLE = False

MIN_ROWS_FOR_TRAINING = 30
MAX_ONE_HOT_CARDINALITY = 20


def _detect_problem_type(target: pd.Series) -> str:
    """Return 'classification' or 'regression' for the given target series."""
    if pd.api.types.is_bool_dtype(target):
        return "classification"

    if pd.api.types.is_numeric_dtype(target):
        n_unique = target.nunique(dropna=True)
        # A numeric column with very few distinct values is almost
        # certainly an encoded class label (e.g. 0/1 churn flag), not a
        # continuous regression target.
        if n_unique <= 10 and n_unique / max(len(target), 1) < 0.05:
            return "classification"
        return "regression"

    return "classification"


def _prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric feature matrix via one-hot encoding of low-cardinality
    categoricals, dropping high-cardinality / identifier-like columns and
    the target itself.
    """
    feature_df = df.drop(columns=[target_column]).copy()
    usable_columns: list[str] = []

    for col in feature_df.columns:
        series = feature_df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            usable_columns.append(col)
        elif pd.api.types.is_bool_dtype(series):
            usable_columns.append(col)
        elif series.nunique(dropna=True) <= MAX_ONE_HOT_CARDINALITY:
            usable_columns.append(col)
        # else: high-cardinality string/id-like column -- dropped, since
        # one-hot encoding it would explode dimensionality without adding
        # real predictive signal for a baseline model.

    if not usable_columns:
        raise InsufficientDataError(
            "No usable feature columns found (all columns were either the "
            "target, or high-cardinality identifier-like text columns)."
        )

    working = feature_df[usable_columns]
    numeric_part = working.select_dtypes(include="number")
    categorical_part = working.select_dtypes(exclude="number")

    encoded_categorical = (
        pd.get_dummies(categorical_part, dummy_na=False) if not categorical_part.empty else pd.DataFrame(index=working.index)
    )

    features = pd.concat([numeric_part, encoded_categorical], axis=1)
    features = features.fillna(features.median(numeric_only=True)).fillna(0)

    return features, list(features.columns)


def _build_classification_models(use_xgboost: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if use_xgboost and _XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss", use_label_encoder=False
        )
    return models


def _build_regression_models(use_xgboost: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if use_xgboost and _XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(n_estimators=200, random_state=42)
    return models


def train_baseline_model(
    df: pd.DataFrame,
    target_column: str,
    use_xgboost: bool = True,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train baseline models for the given target, auto-detecting
    classification vs regression, and return metrics for each.

    Returns a dict with `problem_type`, `models` (per-model metrics), the
    identified `best_model`, and enough metadata (feature names, encoders,
    the fitted best estimator + processed feature matrix) for downstream
    tools (feature importance, SHAP) to reuse without retraining.
    """
    if target_column not in df.columns:
        raise InvalidToolArgumentsError(f"Target column '{target_column}' not found in dataset.")

    working = df.dropna(subset=[target_column]).copy()
    if len(working) < MIN_ROWS_FOR_TRAINING:
        raise InsufficientDataError(
            f"Only {len(working)} rows have a non-missing target value; at least "
            f"{MIN_ROWS_FOR_TRAINING} are required to train a baseline model."
        )

    problem_type = _detect_problem_type(working[target_column])

    target_encoder: LabelEncoder | None = None
    if problem_type == "classification":
        raw_target = working[target_column].astype(str)
        n_classes = raw_target.nunique()
        if n_classes < 2:
            raise ModelTrainingError(
                f"Target column '{target_column}' contains only one class "
                f"('{raw_target.iloc[0]}'); a classification model needs at least two."
            )
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(raw_target)
    else:
        y = working[target_column].astype(float).values

    X, feature_names = _prepare_features(working, target_column)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if problem_type == "classification" else None,
        )
    except ValueError as exc:
        raise ModelTrainingError(
            f"Could not create a train/test split (often caused by a class with too "
            f"few members for stratified splitting): {exc}"
        ) from exc

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = (
        _build_classification_models(use_xgboost)
        if problem_type == "classification"
        else _build_regression_models(use_xgboost)
    )

    results: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in models.items():
        try:
            # Linear models benefit from scaling; tree ensembles don't need it
            # but are unaffected by it, so using scaled features uniformly
            # keeps this loop simple.
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            if problem_type == "classification":
                metrics: dict[str, float] = {
                    "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                    "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
                    "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
                }
                if len(np.unique(y_train)) == 2 and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test_scaled)[:, 1]
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
            else:
                metrics = {
                    "r2": round(float(r2_score(y_test, preds)), 4),
                    "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                    "rmse": round(float(np.sqrt(np.mean((y_test - preds) ** 2))), 4),
                }

            results[name] = metrics
            fitted_models[name] = model
        except Exception as exc:  # noqa: BLE001 - one model failing shouldn't kill the whole tool
            logger.warning("Model '%s' failed to train: %s", name, exc)
            results[name] = {"error": str(exc)}

    if not fitted_models:
        raise ModelTrainingError("All candidate baseline models failed to train.")

    # Pick the best model by the primary metric for the problem type.
    primary_metric = "roc_auc" if problem_type == "classification" else "r2"
    scored = {
        name: metrics.get(primary_metric, metrics.get("accuracy" if problem_type == "classification" else "r2", -np.inf))
        for name, metrics in results.items()
        if "error" not in metrics
    }
    best_model_name = max(scored, key=scored.get)

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "feature_names": feature_names,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": results,
        "best_model": best_model_name,
        # Internal-only fields (not sent to the LLM directly, but used by
        # downstream tools such as SHAP / feature importance within the
        # same agent run).
        "_fitted_models": fitted_models,
        "_scaler": scaler,
        "_X_test_scaled": X_test_scaled,
        "_X_test_raw": X_test,
        "_target_encoder": target_encoder,
        "_target_classes": list(target_encoder.classes_) if target_encoder is not None else None,
    }


def calculate_feature_importance(train_result: dict[str, Any], top_n: int = 15) -> dict[str, Any]:
    """Model-based feature importance for the best model from
    `train_baseline_model`'s output.
    """
    best_name = train_result["best_model"]
    model = train_result["_fitted_models"][best_name]
    feature_names = train_result["feature_names"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        raise ModelTrainingError(f"Model '{best_name}' does not expose feature importances.")

    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)[:top_n]
    return {
        "model": best_name,
        "top_features": [{"feature": name, "importance": round(float(val), 4)} for name, val in pairs],
    }


def detect_anomalies(df: pd.DataFrame, columns: list[str] | None = None, contamination: float = 0.05) -> dict[str, Any]:
    """Isolation Forest anomaly detection over numeric columns.

    Returns the number/percentage of flagged anomalies and a small sample
    of the most anomalous rows (as index + score), without dumping the
    entire dataset back to the LLM.
    """
    numeric_df = df[columns].select_dtypes(include="number") if columns else df.select_dtypes(include="number")
    numeric_df = numeric_df.dropna()

    if numeric_df.shape[1] < 1:
        raise InvalidToolArgumentsError("No numeric columns available for anomaly detection.")
    if len(numeric_df) < MIN_ROWS_FOR_TRAINING:
        raise InsufficientDataError(
            f"Only {len(numeric_df)} complete numeric rows available; at least "
            f"{MIN_ROWS_FOR_TRAINING} are required for anomaly detection."
        )

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds = model.fit_predict(numeric_df)
    scores = model.score_samples(numeric_df)

    anomaly_mask = preds == -1
    n_anomalies = int(anomaly_mask.sum())

    anomalous_rows = (
        numeric_df.loc[anomaly_mask]
        .assign(anomaly_score=scores[anomaly_mask])
        .sort_values("anomaly_score")
        .head(10)
    )

    return {
        "columns_used": list(numeric_df.columns),
        "n_rows_analyzed": int(len(numeric_df)),
        "n_anomalies": n_anomalies,
        "anomaly_pct": round(n_anomalies / len(numeric_df) * 100, 2),
        "top_anomalies": anomalous_rows.reset_index().rename(columns={"index": "row_index"}).to_dict(orient="records"),
    }

"""SHAP explanation tool (Section 6, tool #9: `calculate_shap`).

Operates on the fitted model/feature matrix produced by
`train_baseline_model` (Section 6, tool #7), so SHAP never needs to
retrain anything. Returns global feature importance and top per-feature
contribution summaries -- never raw per-row SHAP arrays, to keep the
payload sent back to the LLM small (Section 5/8).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shap

from app.core.exceptions import ModelTrainingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Cap the background/explanation sample size for speed -- SHAP on tree
# ensembles is fast, but this keeps behavior predictable regardless of
# dataset size, and this is a baseline/demo tool, not a production
# explainability pipeline.
MAX_SHAP_SAMPLE_ROWS = 200


def _direction_from_correlation(feature_values: np.ndarray, shap_col: np.ndarray) -> tuple[str, float]:
    """Determine a feature's direction of effect via the correlation
    between its value and its SHAP value across samples -- the same
    convention SHAP's own summary/beeswarm plots use (coloring by
    feature value) to describe "high values push the prediction up/down".

    This is used instead of the mean *signed* SHAP value across all rows,
    which is misleading for one-hot/dummy features: averaging a strong
    positive contribution (when the dummy is 1) together with a mild
    negative contribution (when the dummy is 0) can flip the overall sign
    depending on class balance, even when the true effect ("this category
    increases the prediction when present") is unambiguous.
    """
    if np.std(feature_values) == 0 or np.std(shap_col) == 0:
        return "no consistent direction (feature has no variance in this sample)", 0.0
    corr = float(np.corrcoef(feature_values, shap_col)[0, 1])
    if np.isnan(corr):
        return "no consistent direction (feature has no variance in this sample)", 0.0
    direction = (
        "higher values of this feature increase the prediction"
        if corr > 0
        else "higher values of this feature decrease the prediction"
    )
    return direction, round(corr, 4)


def calculate_shap(train_result: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
    """Compute SHAP values for the best model from `train_baseline_model`.

    Returns:
        - `model`: which model was explained
        - `top_features`: mean absolute SHAP value per feature (global
          importance), plus a `direction` derived from the correlation
          between the feature's value and its SHAP value across samples
          (see `_direction_from_correlation`) -- not a raw average, which
          can be misleading for one-hot/dummy features.
    """
    best_name = train_result["best_model"]
    model = train_result["_fitted_models"][best_name]
    feature_names = train_result["feature_names"]
    X_test_scaled = train_result["_X_test_scaled"]

    sample = X_test_scaled[:MAX_SHAP_SAMPLE_ROWS]

    try:
        if hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, XGBoost): TreeExplainer is
            # fast and exact.
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            # For binary classifiers, shap_values may be a list [class0, class1].
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            # Linear models: LinearExplainer is fast and exact.
            explainer = shap.LinearExplainer(model, sample)
            shap_values = explainer.shap_values(sample)
    except Exception as exc:  # noqa: BLE001
        raise ModelTrainingError(f"SHAP explanation failed for model '{best_name}': {exc}") from exc

    shap_values = np.asarray(shap_values)
    if shap_values.ndim > 2:
        # Some SHAP/model version combinations return shape
        # (n_samples, n_features, n_classes); take the positive class.
        shap_values = shap_values[:, :, -1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:top_n]

    top_features = []
    for idx in order:
        direction, correlation = _direction_from_correlation(sample[:, idx], shap_values[:, idx])
        top_features.append(
            {
                "feature": feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                "value_shap_correlation": correlation,
                "direction": direction,
            }
        )

    return {
        "model": best_name,
        "n_samples_explained": int(len(sample)),
        "top_features": top_features,
    }

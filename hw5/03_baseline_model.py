"""
Baseline model trained on the full (post-cleaning) feature set.

Objectives of the script:
1. Train XGBoost on unbalanced data "as-is."
2. Measure baseline metrics (PR-AUC, Recall, Precision, F1 on the fraud class).
3. Select the top-K features by importance -> these will be fed into CTGAN
   (training a GAN on 400 columns is unstable and time-consuming, so we reduce
   the feature space to the most informative features).
"""
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from config import (
    TARGET_COL, TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH,
    XGB_PARAMS, TOP_K, TOP_K_FEATURES_PATH, FULL_MODEL_PATH, RANDOM_STATE,
)


def load_split():
    train_df = pd.read_parquet(TRAIN_PROCESSED_PATH)
    test_df = pd.read_parquet(TEST_PROCESSED_PATH)
    return train_df, test_df


def evaluate(model, X_test, y_test, label="model"):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds)

    print(f"\n=== Metrics: {label} ===")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"ROC-AUC:                    {roc_auc:.4f}")
    print(f"Precision (fraud):           {precision:.4f}")
    print(f"Recall (fraud):               {recall:.4f}")
    print(f"F1 (fraud):                   {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(classification_report(y_test, preds, target_names=["Normal", "Fraud"], zero_division=0))

    return dict(pr_auc=pr_auc, roc_auc=roc_auc, precision=precision, recall=recall, f1=f1)


def main():
    train_df, test_df = load_split()

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    print(f"[baseline] Training on {X_train.shape[1]} features, {X_train.shape[0]} rows")

    # scale_pos_weight helps XGBoost account for class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"[baseline] scale_pos_weight = {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test, label="Baseline (full feature set)")

    joblib.dump(model, FULL_MODEL_PATH)
    print(f"[baseline] Model saved: {FULL_MODEL_PATH}")

    # ---- Feature Selection for GAN ----
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_k_features = importances.sort_values(ascending=False).head(TOP_K).index.tolist()

    print(f"\n[baseline] Top-{TOP_K} features by importance:")
    for i, f in enumerate(top_k_features, 1):
        print(f"  {i:2d}. {f}  (importance={importances[f]:.4f})")

    with open(TOP_K_FEATURES_PATH, "w") as f:
        json.dump(top_k_features, f, ensure_ascii=False, indent=2)
    print(f"\n[baseline] The list of the top-K features has been saved: {TOP_K_FEATURES_PATH}")

    with open(FULL_MODEL_PATH.parent / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
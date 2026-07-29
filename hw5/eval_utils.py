"""
General utilities for evaluating models.

Here, we find the threshold that maximizes F1 on the PR curve and calculate the metrics
at that threshold—this ensures that comparisons between models are valid.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_recall_fscore_support, 
)


def find_best_threshold(y_true, proba, metric="f1"):
    """Finds the threshold on the PR curve that maximizes the specified metric (default is F1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more element than thresholds -> trim
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_with_best_threshold(y_true, proba, label="model", verbose=True):
    """Full evaluation of the model: PR-AUC/ROC-AUC (independent of threshold) +
    Precision/Recall/F1 at the optimal threshold (maximizing F1)."""
    pr_auc = average_precision_score(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)

    best_threshold, _ = find_best_threshold(y_true, proba)
    preds = (proba >= best_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )

    if verbose:
        print(f"\n=== {label} (threshold chosen to maximize F1 = {best_threshold:.4f}) ===")
        print(f"PR-AUC:    {pr_auc:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")

    return dict(
        strategy=label, pr_auc=pr_auc, roc_auc=roc_auc,
        precision=precision, recall=recall, f1=f1,
        best_threshold=float(best_threshold),
    )


def get_pr_curve(y_true, proba):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return precision, recall

"""
Final comparison of strategies for addressing class imbalance,
all using the same set of top-K features (fair comparison):

1. Baseline              — no balancing, only `scale_pos_weight`.
2. SMOTE                 — classic oversampling via interpolation.
3. GAN-augmented         — real data + synthetic data from CTGAN (default hyperparameters).
4. GAN-augmented (tuned) — same as above, but with synthetic data from CTGAN tuned by Optuna
                            (used only if 08_tune_ctgan_optuna.py is running).

"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from config import (
    TARGET_COL, TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH,
    TOP_K_FEATURES_PATH, SYNTHETIC_FRAUD_PATH, TUNED_SYNTHETIC_FRAUD_PATH,
    XGB_BEST_PARAMS_PATH, PLOTS_DIR, RANDOM_STATE,
)
from eval_utils import evaluate_with_best_threshold, get_pr_curve


def load_xgb_params():
    """If Optuna-tuning of XGBoost was run (07_tune_xgboost_optuna.py) —
    use the found parameters, otherwise — use default parameters."""
    if XGB_BEST_PARAMS_PATH.exists():
        with open(XGB_BEST_PARAMS_PATH) as f:
            params = json.load(f)
        print(f"[compare] Using Optuna-tuned XGBoost parameters from {XGB_BEST_PARAMS_PATH}")
        return params
    print("[compare] Optuna-tuned XGBoost parameters not found, using default.")
    return dict(n_estimators=300, max_depth=6, learning_rate=0.05)


def fit_model(X_train, y_train, extra_params=None):
    params = dict(eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)
    params.update(load_xgb_params())
    if extra_params:
        params.update(extra_params)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def main():
    train_df = pd.read_parquet(TRAIN_PROCESSED_PATH)
    test_df = pd.read_parquet(TEST_PROCESSED_PATH)

    with open(TOP_K_FEATURES_PATH) as f:
        features = json.load(f)

    X_train_full, y_train_full = train_df[features], train_df[TARGET_COL]
    X_test, y_test = test_df[features], test_df[TARGET_COL]

    results = []
    pr_curves = {}  # label -> (precision, recall) for the final plot

    # ---- 1. Baseline ----
    neg, pos = (y_train_full == 0).sum(), (y_train_full == 1).sum()
    model_baseline = fit_model(X_train_full, y_train_full, extra_params=dict(scale_pos_weight=neg / pos))
    proba = model_baseline.predict_proba(X_test)[:, 1]
    results.append(evaluate_with_best_threshold(y_test, proba, label="Baseline"))
    pr_curves["Baseline"] = get_pr_curve(y_test, proba)

    # ---- 2. SMOTE ----
    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_train_full, y_train_full)
    model_smote = fit_model(X_smote, y_smote)
    proba = model_smote.predict_proba(X_test)[:, 1]
    results.append(evaluate_with_best_threshold(y_test, proba, label="SMOTE"))
    pr_curves["SMOTE"] = get_pr_curve(y_test, proba)

    # ---- 3. GAN-augmented (default CTGAN hyperparameters) ----
    if SYNTHETIC_FRAUD_PATH.exists():
        synthetic_fraud = pd.read_parquet(SYNTHETIC_FRAUD_PATH)
        gan_train = pd.concat([
            train_df[features + [TARGET_COL]],
            synthetic_fraud[features + [TARGET_COL]],
        ], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
        model_gan = fit_model(gan_train[features], gan_train[TARGET_COL])
        proba = model_gan.predict_proba(X_test)[:, 1]
        results.append(evaluate_with_best_threshold(y_test, proba, label="GAN-augmented (CTGAN)"))
        pr_curves["GAN-augmented (CTGAN)"] = get_pr_curve(y_test, proba)
    else:
        print("[compare] Synthetic data from the baseline CTGAN not found, skipping (run 04_ctgan_synthesis.py).")

    # ---- 4. GAN-augmented (tuned via Optuna CTGAN), if available ----
    if TUNED_SYNTHETIC_FRAUD_PATH.exists():
        synthetic_fraud_tuned = pd.read_parquet(TUNED_SYNTHETIC_FRAUD_PATH)
        gan_train_tuned = pd.concat([
            train_df[features + [TARGET_COL]],
            synthetic_fraud_tuned[features + [TARGET_COL]],
        ], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
        model_gan_tuned = fit_model(gan_train_tuned[features], gan_train_tuned[TARGET_COL])
        proba = model_gan_tuned.predict_proba(X_test)[:, 1]
        results.append(evaluate_with_best_threshold(y_test, proba, label="GAN-augmented (CTGAN, Optuna-tuned)"))
        pr_curves["GAN-augmented (CTGAN, Optuna-tuned)"] = get_pr_curve(y_test, proba)
    else:
        print("[compare] Tuned synthetic data not found, skipping "
              "(run 08_tune_ctgan_optuna.py if you want to compare).")

    # ---- Summary Table ----
    results_df = pd.DataFrame(results)
    print("\n=== Summary Table (Metrics at Optimal Threshold for Each Model) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(PLOTS_DIR.parent / "final_comparison.csv", index=False)

    # ---- Plot 1: metrics by strategies ----
    metrics_to_plot = ["pr_auc", "recall", "precision", "f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(results_df))
    width = 0.2
    for i, metric in enumerate(metrics_to_plot):
        ax.bar([p + i * width for p in x], results_df[metric], width=width, label=metric)
    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels(results_df["strategy"], rotation=15, ha="right")
    ax.set_ylabel("Value of the Metric")
    ax.set_title("Comparison of Strategies (Optimal Threshold for Each Model)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_strategy_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: PR-curves for all strategies on a single plot ----
    # This is fairer than the metrics plot with a fixed threshold: it shows the entire
    # precision/recall trade-off, not just one point.
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (precision, recall) in pr_curves.items():
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-curves: comparison of strategies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_pr_curves_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\n[compare] Graphs: {PLOTS_DIR / '07_strategy_comparison.png'}, "
          f"{PLOTS_DIR / '08_pr_curves_comparison.png'}")
    print(f"[compare] Metrics Table: {PLOTS_DIR.parent / 'final_comparison.csv'}")


if __name__ == "__main__":
    main()

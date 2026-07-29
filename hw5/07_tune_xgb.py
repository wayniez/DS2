"""
Tuning XGBoost hyperparameters using Optuna.

We optimize PR-AUC (average precision) on hold-out validation—
this is the primary metric for highly imbalanced classes, unlike
accuracy or even ROC-AUC.

Validation is time-based (just like the test set), meaning the model cannot see
the future with respect to the validation data—this is important for a fair evaluation
on data with a temporal structure (transactions).
"""
import json
import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from config import (
    TARGET_COL, TIME_COL, TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH,
    N_OPTUNA_TRIALS_XGB, OPTUNA_VAL_SIZE, XGB_BEST_PARAMS_PATH,
    FULL_MODEL_PATH, RANDOM_STATE,
)
from eval_utils import evaluate_with_best_threshold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def time_based_val_split(train_df: pd.DataFrame, val_size: float):
    df_sorted = train_df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_size))
    return df_sorted.iloc[:split_idx].reset_index(drop=True), df_sorted.iloc[split_idx:].reset_index(drop=True)


def make_objective(X_train, y_train, X_val, y_val, scale_pos_weight):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 700),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            # We also tune `scale_pos_weight` around the "theoretical" value —
            # sometimes the model performs better with a less aggressive imbalance correction
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight", scale_pos_weight * 0.3, scale_pos_weight * 1.5
            ),
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    return objective


def main():
    train_df = pd.read_parquet(TRAIN_PROCESSED_PATH)
    test_df = pd.read_parquet(TEST_PROCESSED_PATH)

    feature_cols = [c for c in train_df.columns if c not in (TARGET_COL, TIME_COL)]

    train_sub, val_sub = time_based_val_split(train_df, OPTUNA_VAL_SIZE)
    print(f"[optuna-xgb] train_sub: {train_sub.shape}, val_sub: {val_sub.shape}")

    X_train, y_train = train_sub[feature_cols], train_sub[TARGET_COL]
    X_val, y_val = val_sub[feature_cols], val_sub[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    study = optuna.create_study(direction="maximize", study_name="xgb_fraud_pr_auc")
    objective = make_objective(X_train, y_train, X_val, y_val, scale_pos_weight)

    print(f"[optuna-xgb] Running {N_OPTUNA_TRIALS_XGB} trials ...")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS_XGB, show_progress_bar=True)

    print(f"\n[optuna-xgb] Best PR-AUC on validation: {study.best_value:.4f}")
    print("[optuna-xgb] Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open(XGB_BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[optuna-xgb] Parameters saved: {XGB_BEST_PARAMS_PATH}")

    # ---- Retrain the final model on the entire training set with the best parameters ----
    best_params = dict(study.best_params)
    best_params.update(eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)

    X_train_full, y_train_full = train_df[feature_cols], train_df[TARGET_COL]
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_full, y_train_full)

    proba_test = final_model.predict_proba(X_test)[:, 1]
    evaluate_with_best_threshold(y_test, proba_test, label="XGBoost (Optuna-tuning)")

    joblib.dump(final_model, FULL_MODEL_PATH.parent / "xgb_optuna_tuned.joblib")
    print(f"[optuna-xgb] Final model saved: "
          f"{FULL_MODEL_PATH.parent / 'xgb_optuna_tuned.joblib'}")


if __name__ == "__main__":
    main()

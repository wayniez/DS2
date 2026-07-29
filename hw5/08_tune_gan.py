"""
Hyperparameter tuning for CTGAN using Optuna.

This is the most valuable (and most expensive) tuning step in the project: the quality
of the synthetic data directly depends on the generator/discriminator architecture,
and this was likely the cause of the low recall in the GAN-augmented
model (see the discussion in the README)—the synthetic data was not “diverse” enough.

"""
import json
import joblib
import optuna
import pandas as pd
import xgboost as xgb
from ctgan import CTGAN
from sklearn.metrics import average_precision_score

from config import (
    TARGET_COL, TIME_COL, TRAIN_PROCESSED_PATH, TOP_K_FEATURES_PATH,
    KNOWN_CATEGORICAL_COLS, N_OPTUNA_TRIALS_CTGAN, CTGAN_TUNING_EPOCHS,
    CTGAN_EPOCHS, OPTUNA_VAL_SIZE, CTGAN_BEST_PARAMS_PATH,
    TUNED_CTGAN_MODEL_PATH, TUNED_SYNTHETIC_FRAUD_PATH, RANDOM_STATE,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def time_based_val_split(train_df: pd.DataFrame, val_size: float):
    df_sorted = train_df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_size))
    return df_sorted.iloc[:split_idx].reset_index(drop=True), df_sorted.iloc[split_idx:].reset_index(drop=True)


def tstr_pr_auc(real_normal, synthetic_fraud, val_df, features):
    """Quick TSTR Evaluation: Training a Simple XGBoost Model and Restoring the PR-AUC on the Validation Set."""
    train_data = pd.concat([
        real_normal[features + [TARGET_COL]],
        synthetic_fraud[features + [TARGET_COL]],
    ], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train_data[features], train_data[TARGET_COL])

    proba = model.predict_proba(val_df[features])[:, 1]
    return average_precision_score(val_df[TARGET_COL], proba)


def make_objective(fraud_df, real_normal, val_df, features, categorical_in_topk, n_fraud):
    def objective(trial: optuna.Trial) -> float:
        # pac должен быть делителем batch_size (ограничение CTGAN) -> подбираем совместно
        batch_size = trial.suggest_categorical("batch_size", [100, 200, 500])
        pac = trial.suggest_categorical("pac", [1, 2, 5, 10])
        if batch_size % pac != 0:
            raise optuna.TrialPruned()

        gen_dim_choice = trial.suggest_categorical("generator_dim", ["small", "medium", "large"])
        dim_map = {
            "small": (128, 128),
            "medium": (256, 256),
            "large": (256, 256, 256),
        }
        generator_dim = dim_map[gen_dim_choice]
        disc_dim_choice = trial.suggest_categorical("discriminator_dim", ["small", "medium", "large"])
        discriminator_dim = dim_map[disc_dim_choice]

        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        discriminator_steps = trial.suggest_int("discriminator_steps", 1, 5)
        generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)

        ctgan = CTGAN(
            epochs=CTGAN_TUNING_EPOCHS,
            batch_size=batch_size,
            pac=pac,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            embedding_dim=embedding_dim,
            discriminator_steps=discriminator_steps,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            verbose=False,
            cuda=True,
        )
        ctgan.fit(fraud_df, discrete_columns=categorical_in_topk)

        synthetic_fraud = ctgan.sample(n_fraud)
        synthetic_fraud[TARGET_COL] = 1

        score = tstr_pr_auc(real_normal, synthetic_fraud, val_df, features)
        return score

    return objective


def main():
    train_df = pd.read_parquet(TRAIN_PROCESSED_PATH)
    with open(TOP_K_FEATURES_PATH) as f:
        features = json.load(f)

    train_sub, val_sub = time_based_val_split(train_df, OPTUNA_VAL_SIZE)
    print(f"[optuna-ctgan] train_sub: {train_sub.shape}, val_sub: {val_sub.shape}")

    fraud_df = train_sub.loc[train_sub[TARGET_COL] == 1, features].reset_index(drop=True)
    real_normal = train_sub.loc[train_sub[TARGET_COL] == 0, features + [TARGET_COL]]
    n_fraud = len(fraud_df)
    print(f"[optuna-ctgan] Fraud examples for GAN training: {n_fraud}")

    categorical_in_topk = [c for c in features if c in KNOWN_CATEGORICAL_COLS]

    study = optuna.create_study(direction="maximize", study_name="ctgan_tstr_pr_auc")
    objective = make_objective(fraud_df, real_normal, val_sub, features, categorical_in_topk, n_fraud)

    print(f"[optuna-ctgan] Running {N_OPTUNA_TRIALS_CTGAN} trials "
          f"(each = training CTGAN on {CTGAN_TUNING_EPOCHS} epochs) ...")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS_CTGAN, show_progress_bar=True)

    print(f"\n[optuna-ctgan] Best TSTR PR-AUC: {study.best_value:.4f}")
    print("[optuna-ctgan] Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open(CTGAN_BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[optuna-ctgan] Best parameters saved: {CTGAN_BEST_PARAMS_PATH}")

    # ---- Retrain the final CTGAN on the FULL training set with the best parameters
    #      and the full number of epochs (CTGAN_EPOCHS, not shortened) ----
    dim_map = {"small": (128, 128), "medium": (256, 256), "large": (256, 256, 256)}
    bp = study.best_params

    full_fraud_df = train_df.loc[train_df[TARGET_COL] == 1, features].reset_index(drop=True)
    n_normal_full = (train_df[TARGET_COL] == 0).sum()

    final_ctgan = CTGAN(
        epochs=CTGAN_EPOCHS,
        batch_size=bp["batch_size"],
        pac=bp["pac"],
        generator_dim=dim_map[bp["generator_dim"]],
        discriminator_dim=dim_map[bp["discriminator_dim"]],
        embedding_dim=bp["embedding_dim"],
        discriminator_steps=bp["discriminator_steps"],
        generator_lr=bp["generator_lr"],
        discriminator_lr=bp["discriminator_lr"],
        verbose=True,
        cuda=True,
    )
    print("\n[optuna-ctgan] Training the final CTGAN on the full training set with the best parameters ...")
    final_ctgan.fit(full_fraud_df, discrete_columns=categorical_in_topk)

    joblib.dump(final_ctgan, TUNED_CTGAN_MODEL_PATH)

    n_to_generate = n_normal_full - len(full_fraud_df)
    synthetic_fraud_tuned = final_ctgan.sample(n_to_generate)
    synthetic_fraud_tuned[TARGET_COL] = 1
    synthetic_fraud_tuned.to_parquet(TUNED_SYNTHETIC_FRAUD_PATH, index=False)

    print(f"[optuna-ctgan] Tuned CTGAN saved: {TUNED_CTGAN_MODEL_PATH}")
    print(f"[optuna-ctgan] Tuned synthetic fraud saved: {TUNED_SYNTHETIC_FRAUD_PATH}")


if __name__ == "__main__":
    main()

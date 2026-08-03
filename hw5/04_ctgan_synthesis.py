"""
Training the CTGAN on a subset of fraudulent transactions (top-K features)
and generating synthetic fraudulent transactions.

Why we train the GAN only on the fraud class, rather than on the entire dataset:
- Our goal is specifically to augment the rare class, not to generate
  any transactions at all;
- Training a conditional GAN on both classes with a condition on `isFraud` -
  a valid alternative approach, but it requires more data and epochs for
  stable convergence; for a thesis, the option of "GAN trained only on
  the fraud subset" is easier to explain and reproduce.
"""
import json
import joblib
import pandas as pd
from ctgan import CTGAN

from config import (
    TARGET_COL, TRAIN_PROCESSED_PATH, TOP_K_FEATURES_PATH,
    KNOWN_CATEGORICAL_COLS, CTGAN_EPOCHS, CTGAN_BATCH_SIZE,
    CTGAN_MODEL_PATH, SYNTHETIC_FRAUD_PATH, N_SYNTHETIC_SAMPLES,
    RANDOM_STATE,
)


def main():
    train_df = pd.read_parquet(TRAIN_PROCESSED_PATH)

    with open(TOP_K_FEATURES_PATH) as f:
        top_k_features = json.load(f)

    fraud_df = train_df.loc[train_df[TARGET_COL] == 1, top_k_features].reset_index(drop=True)
    n_normal = (train_df[TARGET_COL] == 0).sum()
    n_fraud = len(fraud_df)

    print(f"[ctgan] Training CTGAN on {n_fraud} fraudulent transactions, {len(top_k_features)} features")

    categorical_in_topk = [c for c in top_k_features if c in KNOWN_CATEGORICAL_COLS]
    print(f"[ctgan] Categorical features among top-K: {len(categorical_in_topk)} -> {categorical_in_topk}")

    ctgan = CTGAN(
        epochs=CTGAN_EPOCHS,
        batch_size=CTGAN_BATCH_SIZE,
        verbose=True,
        enable_gpu=True,  # CTGAN automatically falls back to the CPU if CUDA is unavailable
    )

    ctgan.fit(fraud_df, discrete_columns=categorical_in_topk)

    joblib.dump(ctgan, CTGAN_MODEL_PATH)
    print(f"[ctgan] Model CTGAN saved: {CTGAN_MODEL_PATH}")

    # How many synthetics to generate: by default - balance the number
    # of fraud examples with the number of normal transactions (full balancing train)
    n_to_generate = N_SYNTHETIC_SAMPLES or (n_normal - n_fraud)
    n_to_generate = max(n_to_generate, n_fraud)  # safety margin

    print(f"[ctgan] Generating {n_to_generate} synthetic fraudulent transactions ...")
    synthetic_fraud = ctgan.sample(n_to_generate)
    synthetic_fraud[TARGET_COL] = 1

    synthetic_fraud.to_parquet(SYNTHETIC_FRAUD_PATH, index=False)
    print(f"[ctgan] Synthetic data saved: {SYNTHETIC_FRAUD_PATH}")
    print(synthetic_fraud.head())


if __name__ == "__main__":
    main()
"""
Data preprocessing:
1. Drop columns with a missing value rate higher than MISSING_THRESHOLD.
2. Imputation (numeric → median, categorical → separate "missing" category).
3. Encoding categorical features (label encoding, since we’ll be using XGBoost later).
4. Time-based train/test split using TransactionDT (closer to a real-world scenario
   than a random split: the model is trained on “past” data and tested on “future” data).
5. Saving the processed train/test datasets in Parquet.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    TARGET_COL, ID_COL, TIME_COL, MISSING_THRESHOLD,
    KNOWN_CATEGORICAL_COLS, TEST_SIZE,
    TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH, MODELS_DIR,
)
from data_loading import load_raw


def drop_high_missing_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    print(f"[preprocessing] Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing values")
    return df.drop(columns=cols_to_drop)


def encode_categoricals(df: pd.DataFrame, categorical_cols: list) -> tuple[pd.DataFrame, dict]:
    """Label encoding categorical features. Missing values -> separate category."""
    encoders = {}
    df = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    return df, encoders


def fill_numeric_missing(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def time_based_split(df: pd.DataFrame, test_size: float):
    """Sort by time, last test_size% rows -> test.
    This is fairer than random split for fraud detection: the model doesn't peek into the future."""
    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def main():
    df = load_raw()

    df = drop_high_missing_columns(df, MISSING_THRESHOLD)

    # We don't need the transaction ID as a feature
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    categorical_cols = [c for c in KNOWN_CATEGORICAL_COLS if c in df.columns]
    numeric_cols = [
        c for c in df.columns
        if c not in categorical_cols + [TARGET_COL, TIME_COL]
    ]

    print(f"[preprocessing] Categorical features: {len(categorical_cols)}")
    print(f"[preprocessing] Numeric features: {len(numeric_cols)}")

    df = fill_numeric_missing(df, numeric_cols)
    df, encoders = encode_categoricals(df, categorical_cols)

    # Save the encoders for later (e.g., if we want to decode the synthetic data back into a human-readable format)
    # the synthetic data back into a human-readable format)
    with open(MODELS_DIR / "categorical_encoders.json", "w") as f:
        json.dump(encoders, f)

    train_df, test_df = time_based_split(df, TEST_SIZE)

    print(f"[preprocessing] Train: {train_df.shape}, fraud rate: {train_df[TARGET_COL].mean():.4f}")
    print(f"[preprocessing] Test:  {test_df.shape}, fraud rate: {test_df[TARGET_COL].mean():.4f}")

    train_df.to_parquet(TRAIN_PROCESSED_PATH, index=False)
    test_df.to_parquet(TEST_PROCESSED_PATH, index=False)
    print(f"[preprocessing] Saved {TRAIN_PROCESSED_PATH} and {TEST_PROCESSED_PATH}")


if __name__ == "__main__":
    main()
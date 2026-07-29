"""
Loading raw CSV files from IEEE-CIS and merging transaction + identity data.

train_transaction.csv contains the main transaction information
(including the target variable isFraud).
train_identity.csv contains additional information about the device/session,
but is not available for all transactions (left join via TransactionID).
"""
import pandas as pd
from config import RAW_TRANSACTION_PATH, RAW_IDENTITY_PATH, MERGED_PARQUET_PATH, ID_COL


def load_raw() -> pd.DataFrame:
    """Loads two CSV files, merges them via left join, and caches the result in a parquet file."""
    if MERGED_PARQUET_PATH.exists():
        print(f"[data_loading] Found cache: {MERGED_PARQUET_PATH}, reading it.")
        return pd.read_parquet(MERGED_PARQUET_PATH)

    if not RAW_TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {RAW_TRANSACTION_PATH}. "
        )

    print("[data_loading] Reading train_transaction.csv ...")
    transaction = pd.read_csv(RAW_TRANSACTION_PATH)
    print(f"  -> {transaction.shape}")

    if RAW_IDENTITY_PATH.exists():
        print("[data_loading] Reading train_identity.csv ...")
        identity = pd.read_csv(RAW_IDENTITY_PATH)
        print(f"  -> {identity.shape}")
        df = transaction.merge(identity, on=ID_COL, how="left")
    else:
        print("[data_loading] train_identity.csv not found, working without identity features.")
        df = transaction

    print(f"[data_loading] Final shape: {df.shape}")

    df.to_parquet(MERGED_PARQUET_PATH, index=False)
    print(f"[data_loading] Cached result saved to {MERGED_PARQUET_PATH}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(df["isFraud"].value_counts(normalize=True))
"""
Exploratory Data Analysis (EDA):
- class imbalance
- missing values
- distribution of transaction amounts
- distribution by time / category

All plots are saved in outputs/plots/
"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import PLOTS_DIR, TARGET_COL
from data_loading import load_raw

sns.set_theme(style="whitegrid")


def plot_class_balance(df: pd.DataFrame):
    counts = df[TARGET_COL].value_counts()
    fraud_pct = counts[1] / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index.map({0: "Normal", 1: "Fraud"}), y=counts.values, ax=ax)
    ax.set_title(f"Class Balance (Fraud: {fraud_pct:.2f}%)")
    ax.set_ylabel("Number of Transactions")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_class_balance.png", dpi=150)
    plt.close(fig)
    print(f"[eda] Fraction of fraud: {fraud_pct:.3f}%  (total fraud transactions: {counts[1]})")


def plot_missing_values(df: pd.DataFrame, top_n: int = 30):
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=missing.values, y=missing.index, ax=ax)
    ax.set_title(f"Top-{top_n} Columns by Missing Value Proportion")
    ax.set_xlabel("Fraction of NaN")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_missing_values.png", dpi=150)
    plt.close(fig)
    print(f"[eda] Columns with missing values > 0: {(df.isna().mean() > 0).sum()} out of {df.shape[1]}")


def plot_amount_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(
        df.loc[df[TARGET_COL] == 0, "TransactionAmt"].clip(upper=1000),
        bins=50, ax=axes[0], color="steelblue"
    )
    axes[0].set_title("Transaction Amount - Normal (clipped at 1000)")

    sns.histplot(
        df.loc[df[TARGET_COL] == 1, "TransactionAmt"].clip(upper=1000),
        bins=50, ax=axes[1], color="indianred"
    )
    axes[1].set_title("Transaction Amount - Fraud (clipped at 1000)")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_amount_distribution.png", dpi=150)
    plt.close(fig)


def plot_product_cd(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby("ProductCD")[TARGET_COL].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax)
    ax.set_title("Fraud Rate by ProductCD")
    ax.set_ylabel("Fraud Rate")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_fraud_rate_by_productcd.png", dpi=150)
    plt.close(fig)


def main():
    df = load_raw()

    print("\n=== General Information ===")
    print(f"Size: {df.shape}")
    print(f"Columns: {df.shape[1]}, Rows: {df.shape[0]}")

    plot_class_balance(df)
    plot_missing_values(df)
    plot_amount_distribution(df)
    plot_product_cd(df)

    print(f"\n[eda] Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
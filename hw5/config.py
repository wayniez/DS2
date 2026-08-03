"""
Central project configuration: data paths, constants, hyperparameters.
All scripts import settings from here to avoid duplicating constants.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
SYNTHETIC_DIR = OUTPUTS_DIR / "synthetic"

for d in [MODELS_DIR, PLOTS_DIR, SYNTHETIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"
RAW_IDENTITY_PATH = DATA_DIR / "train_identity.csv"

MERGED_PARQUET_PATH = DATA_DIR / "merged_raw.parquet"
TRAIN_PROCESSED_PATH = DATA_DIR / "train_processed.parquet"
TEST_PROCESSED_PATH = DATA_DIR / "test_processed.parquet"

TOP_K_FEATURES_PATH = MODELS_DIR / "top_k_features.json"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_xgb.joblib"
FULL_MODEL_PATH = MODELS_DIR / "full_features_xgb.joblib"
CTGAN_MODEL_PATH = MODELS_DIR / "ctgan_fraud.pkl"

# Results of Optuna-tuning
XGB_BEST_PARAMS_PATH = MODELS_DIR / "xgb_best_params.json"
CTGAN_BEST_PARAMS_PATH = MODELS_DIR / "ctgan_best_params.json"
TUNED_CTGAN_MODEL_PATH = MODELS_DIR / "ctgan_fraud_tuned.pkl"
TUNED_SYNTHETIC_FRAUD_PATH = SYNTHETIC_DIR / "synthetic_fraud_tuned.parquet"

SYNTHETIC_FRAUD_PATH = SYNTHETIC_DIR / "synthetic_fraud.parquet"

# ---------- Data Constants ----------
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Missing value threshold: columns with a fraction of NaN values above this threshold will be dropped
MISSING_THRESHOLD = 0.70

# Categorical columns in IEEE-CIS (base list; some V-columns and id_*
# are also categorical, but they are numeric-like encoded features)
KNOWN_CATEGORICAL_COLS = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
] + [f"id_{i}" for i in range(12, 39)]  # id_12..id_38 in IEEE-CIS are categorical

# ---------- Business cost (for cost-based threshold) ----------
# Conditional cost of one manual review/blocking of a normal transaction
# (False Positive). In a real project, this number needs to be justified —
# for example, the cost of an analyst's work on one verification, or an estimate
# of losses from customer attrition due to unnecessary card blocks. Here this
# is a demonstration value — easy to change and discuss the sensitivity
# of the result to this parameter (see 06_compare_strategies.py).
REVIEW_COST = 10.0

# ---------- Random state ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # from train

# ---------- Baseline model ----------
TOP_K = 30  # How many features do we include in the "reduced" space for a GAN?

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ---------- Optuna ----------
N_OPTUNA_TRIALS_XGB = 50
N_OPTUNA_TRIALS_CTGAN = 20  # CTGAN trials are computationally expensive (each one = a full GAN training run), so use fewer
CTGAN_TUNING_EPOCHS = 100  # Shorter training during tuning (for speed); final value → CTGAN_EPOCHS
OPTUNA_VAL_SIZE = 0.15  # proportion of the training set set aside for validation during tuning (time-based)

# SQLite storage for Optuna: makes the study persistent—if the process
# is interrupted (Ctrl+C, connection loss, restart), when Optuna is restarted,
# it will resume from the trials that have already been completed, rather than starting from scratch.
OPTUNA_STORAGE_PATH = MODELS_DIR / "optuna_studies.db"
OPTUNA_STORAGE_URL = f"sqlite:///{OPTUNA_STORAGE_PATH}"

# ---------- CTGAN ----------
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 500
# How many synthetic fraud examples to generate.
# By default, the number of synthetic fraud examples is set equal to the number of normal transactions
# in the training set (full balancing). You can set any number manually.
N_SYNTHETIC_SAMPLES = None  # None -> calculated automatically in 04_ctgan_synthesis.py
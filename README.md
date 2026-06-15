# Electricity Cost Prediction

A regression project for predicting electricity costs of buildings based on their characteristics. Three models are trained with hyperparameter tuning via Optuna.

---

## Project Structure

```
├── data/
│   ├── electricity_cost_dataset.csv   # raw dataset
│   ├── electricity_cost_fe.csv        # dataset after feature engineering
│   ├── X.csv                          # features for training
│   └── y.csv                          # target variable
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # exploratory data analysis
│   ├── 02_FE.ipynb                    # feature engineering
│   └── 03_model.ipynb                 # model training and evaluation
│
└── requrements.txt                    # all libraries and their versions used in project
└── README.md
```

---

## Data

**Target variable:** `electricity cost` — electricity cost of a building.

**Features:**

| Feature | Type | Description |
|---|---|---|
| `site area` | numerical | area of the site |
| `water consumption` | numerical | water consumption |
| `recycling rate` | numerical | waste recycling rate |
| `utilisation rate` | numerical | building utilisation rate |
| `air quality index` | numerical | air quality index |
| `issue resolution time` | numerical | time to resolve maintenance issues |
| `resident count` | numerical | number of residents |
| `has residents` | binary | whether the building has residents (created in FE) |
| `type_residential` | binary | building type: residential |
| `type_commercial` | binary | building type: commercial |
| `type_mixed_use` | binary | building type: mixed-use |
| `type_industrial` | binary | building type: industrial |
| `structure_type_enc` | categorical | building type (label encoded) |

---

## Notebooks

### 01_EDA.ipynb — Exploratory Data Analysis

- Loading and initial inspection (shape, dtypes, missing values)
- Distributions of numerical features (histograms, boxplots)
- Outlier detection using the IQR method
- Feature correlation matrix
- Target variable analysis by building type
- Scatter plots: `site area` vs `electricity cost` (r=0.9), `water consumption` vs `electricity cost` (r=0.7)

<img width="1113" height="784" alt="4dd3c5e5-4224-45c8-8774-4f50ada3c13d" src="https://github.com/user-attachments/assets/0d87214b-dd41-403d-a1e8-e0e5e13ae978" />

---

### 02_FE.ipynb — Feature Engineering

- Outlier removal via IQR for `water consumption` and `resident count`
- New binary feature `has residents` (`resident count > 0`)
- One-Hot Encoding of `structure type` → `type_residential`, `type_commercial`, `type_mixed_use`, `type_industrial`
- Label Encoding of `structure type` → `structure_type_enc` (Residential=0, Commercial=1, Mixed-use=2, Industrial=3)
- Correlation analysis of all features with the target variable after FE
- Saving final `X.csv` and `y.csv`

<img width="1384" height="484" alt="image" src="https://github.com/user-attachments/assets/ccb22b97-083b-4ada-b13f-88821462d12f" />

---

### 03_model.ipynb — Modelling

**Train/test split:** 80% train / 20% test, `random_state=42`

**Models:**

- **LightGBM** — gradient boosting, no scaler needed
- **XGBoost** — gradient boosting, no scaler needed
- **ElasticNet** — linear model wrapped in a `Pipeline` with `StandardScaler`

**Hyperparameter tuning:** Optuna, 50 trials per model, 5-fold cross-validation, metric `neg_RMSE`.

<img width="1187" height="495" alt="image" src="https://github.com/user-attachments/assets/e4cc0c7e-cda1-493b-951c-f3b97ae80dc6" />


**Model interpretation:** SHAP `TreeExplainer` for LightGBM and XGBoost — feature importance by mean |SHAP value|.

<img width="789" height="660" alt="image" src="https://github.com/user-attachments/assets/825b16b2-9c2b-42d5-a298-ad3ee202203c" /> <img width="789" height="660" alt="image" src="https://github.com/user-attachments/assets/e8c7de8d-9c9d-4856-9259-4362da98ccb2" />





---

## Installation

```bash
pip install -r "requirements.txt"
```

---

## Results

| Model | RMSE | R² |
|---|---|---|
| LightGBM | 209.46 | 0.9616 |
| XGBoost | 209.83 | 0.9615 |

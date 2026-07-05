# Electricity Cost Prediction

A regression project for predicting electricity costs of buildings based on their characteristics. Three models are trained with hyperparameter tuning via Optuna.

---

## Project Structure

```
├── data/
│   ├── electricity_cost_dataset.csv   # raw dataset
│   ├── electricity_cost_fe.csv        # dataset after feature engineering
│   ├── X_boost.csv                    # features for boosted models
│   ├── X_linear.csv                   # features for linear models
│   └── y.csv                          # target variable
│
├── scripts/
│   ├── 01_EDA.ipynb                   # exploratory data analysis
│   ├── 02_FE.ipynb                    # feature engineering
│   └── 03_model.ipynb                 # model training and evaluation
│
└── requrements.txt                    # all libraries and their versions used in project
└── README.md
```

---

## Data

**Target variable:** `electricity cost` - electricity cost of a building.

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

### 01_EDA.ipynb - Exploratory Data Analysis

- Loading and initial inspection (shape, dtypes, missing values)
- Distributions of numerical features (histograms, boxplots)
- Outlier detection using the IQR method
- Feature correlation matrix
- Target variable analysis by building type
- Scatter plots: `site area` vs `electricity cost` (r=0.9), `water consumption` vs `electricity cost` (r=0.7)

<img width="1096" height="784" alt="image" src="https://github.com/user-attachments/assets/56f411c7-956b-4f77-9db2-4914ec7e4e75" />


---

### 02_FE.ipynb - Feature Engineering

- Outlier removal via IQR for `water consumption` and `resident count`
- New binary feature `has residents` (`resident count > 0`)
- One-Hot Encoding of `structure type` → `type_residential`, `type_commercial`, `type_mixed_use`, `type_industrial`
- Label Encoding of `structure type` → `structure_type_enc` (Residential=0, Commercial=1, Mixed-use=2, Industrial=3)
- Correlation analysis of all features with the target variable after FE
- Saving final `X.csv` and `y.csv`

<img width="1384" height="484" alt="image" src="https://github.com/user-attachments/assets/ccb22b97-083b-4ada-b13f-88821462d12f" />

---

### 03_model.ipynb - Modelling

**Train/test split:** 80% train / 20% test, `random_state=42`

**Models:**

- **LightGBM** - gradient boosting, no scaler needed
- **XGBoost** - gradient boosting, no scaler needed
- **ElasticNet** - linear model wrapped in a `Pipeline` with `StandardScaler`  # excluded from the final comparison due to significantly higher RMSE
- **RandomForest**
- 
**Hyperparameter tuning:** Optuna, 50 trials per model, 5-fold cross-validation, metric `neg_RMSE`.

<img width="1586" height="495" alt="image" src="https://github.com/user-attachments/assets/beb4360a-6994-4af9-8c0b-971a4a2be26a" />




**Model interpretation:** SHAP `TreeExplainer` for LightGBM, XGBoost and Decision Forest- feature importance by mean |SHAP value|.


---

## Installation

```bash
pip install -r "requirements.txt"
```

---

## Results

| Model | RMSE | R² |
|---|---|---|
| LightGBM | 211.26 | 0.9609 |
| XGBoost | 211.09 | 0.9610 |

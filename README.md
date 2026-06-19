# AML Graph Data Mining Starter

Working title:

**Graph-Enhanced Anti-Money Laundering Detection on Enterprise-Scale Synthetic Transaction Data**

This starter repository is organized for a CRISP-DM-based Data Mining final project.

## 1. Dataset

Primary starting point:

- IBM Transactions for Anti-Money Laundering (AML)
- Start with: `HI-Small_Trans.csv`

Put the CSV file here:

```text
data/raw/HI-Small_Trans.csv
```

Expected IBM AMLworld columns:

```text
Timestamp
From Bank
Account
To Bank
Account.1
Amount Received
Receiving Currency
Amount Paid
Payment Currency
Payment Format
Is Laundering
```

The loader also attempts to normalize similar column names automatically.

## 2. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / WSL / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
```

## 3. Notebook order

Run the notebooks in this order:

```text
01_data_understanding.ipynb
02_feature_engineering.ipynb
03_modeling_baseline.ipynb
04_evaluation.ipynb
```

## 4. Development order

Do not start with GNN first.

1. Load and audit the dataset.
2. Build raw transaction features.
3. Train baseline models.
4. Build graph-derived features.
5. Retrain and compare performance.
6. Add graph visualization / deployment.
7. Add GNN only as stretch work.

## 5. Main experiment

Compare these feature sets:

| Experiment | Feature set |
|---|---|
| E1 | Raw transaction features |
| E2 | Raw + temporal behavior features |
| E3 | Raw + temporal + graph-derived features |

Compare these models:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Nonlinear tree baseline |
| XGBoost | Strong gradient boosting |
| LightGBM | Large-data gradient boosting |

Main metrics:

- PR-AUC / Average Precision
- Suspicious-class F1
- Precision
- Recall
- Confusion matrix

## 6. Deployment

The starter `app/streamlit_app.py` expects a trained model in:

```text
models/best_model.joblib
```

Run:

```bash
streamlit run app/streamlit_app.py
```

The first version can be a batch scoring dashboard:
CSV upload → preprocessing → prediction score → ranked suspicious transactions.

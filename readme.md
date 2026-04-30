# CKD Risk Prediction in Diabetic Patients using Longitudinal Data and Machine Learning Approaches

A machine learning pipeline for predicting Chronic Kidney Disease (CKD) onset in diabetic patients using 10-year longitudinal data. The project covers statistical analysis, tree-based models (Random Forest, XGBoost), sequential deep learning models (Vanilla LSTM, BiLSTM+Attention), and SHAP interpretability analysis.

---

## Project Structure

```
.
├── main.py                  # Runs the full pipeline
├── preprocessing.py         # Data cleaning, feature engineering
├── statistical_analysis.py  # Mann-Whitney U, Fisher's Exact, Spearman, Bonferroni/Holm Correction
├── rf_xgboost.py            # Random Forest and XGBoost training + evaluation
├── vanilla_lstm.py          # Vanilla LSTM training + evaluation
├── bilstm_attention.py      # BiLSTM with Attention training + evaluation
├── shap_analysis.py         # SHAP interpretability for XGBoost models
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** [Longitudinal Diabetes Dataset — Mendeley Data V2](https://data.mendeley.com/datasets/hjkzgbxgv5/2)
- 4,000 records · 400 patients · 10-year annual observations
- Download the dataset and place the CSV file in the project root directory before running.

---

## Setup

### 1. Clone or download the project

```bash
cd /path/to/project
```

### 2. Create and activate a conda environment

```bash
conda create -n ckd_prediction python=3.10
conda activate ckd_prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Pipeline

```bash
python main.py
```

This runs all modules in sequence:
1. Data preprocessing and feature engineering
2. Statistical analysis
3. Random Forest and XGBoost modelling
4. Vanilla LSTM modelling
5. BiLSTM+Attention modelling
6. SHAP interpretability analysis

---

## Key Packages

| Package | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and windowing |
| `numpy` | Numerical operations |
| `scikit-learn` | Random Forest, preprocessing, metrics, SMOTE pipeline |
| `xgboost` | XGBoost classifier |
| `imbalanced-learn` | SMOTE oversampling |
| `scipy` | Mann-Whitney U test, Fisher's Exact test, Spearman correlation |
| `statsmodels` | Holm-Bonferroni correction |
| `torch` | Vanilla LSTM and BiLSTM+Attention models |
| `shap` | SHAP value computation and visualisation |
| `matplotlib` | Plotting (ROC, PR curves, feature importance, attention weights) |
| `seaborn` | Heatmaps and statistical plots |

---

## Notes

- Patient-level splitting is used throughout (280 train / 60 val / 60 test) to prevent data leakage.
- Two experimental setups are evaluated: **All Windows** and **Pre-onset Only**.
- Class imbalance is handled via **SMOTE** and **pos_weight scaling** — both strategies are compared per model.
- All plots are written to an `plots/` directory created automatically on first run.

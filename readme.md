# Beyond Prediction: Longitudinal Machine Learning and Causal Inference for CKD Risk in Diabetic Patients

A machine learning pipeline for predicting Chronic Kidney Disease (CKD) onset in diabetic patients using 10-year longitudinal data. The project covers statistical analysis, tree-based models (Random Forest, XGBoost), sequential deep learning models (Vanilla LSTM, BiLSTM+Attention), SHAP interpretability analysis, and causal inference analysis.

---

**Master's capstone, Charles Darwin University (June 2026)**
Group 31: **Sandesh Prasad Paudel**, Sharad Kumar Ranabhat, Orchid Shrestha, Ekramul Hassan Samy
Supervisor: Dr Sami Azam

## My contribution
I was the lead contributor. I designed and implemented the end-to-end data-processing pipeline
(`preprocessing.py`, `utils/`, `main.py`): five-year sliding windows, patient-level 70/15/15 split,
and leakage-safe scaling, SMOTE and threshold selection. I also built all the predictive models and evaluation pipelines:
Random Forest and XGBoost (`rf_xgboost.py`), LSTM (`vanilla_lstm.py`) and BiLSTM with attention (`bilstm_attention.py`).

## Key results
- Five-year sliding windows predict CKD in the following year. The split is by patient (70/15/15), and scaling, SMOTE and thresholds are fitted on training/validation data only.
- Class-weighted XGBoost performed best: ROC-AUC 0.874 with full history, and 0.730 (recall 0.80) when prior CKD history is excluded. LSTM and BiLSTM-attention did not outperform it.
- SHAP: BMI change, diabetes duration and insulin use ranked highest.
- Causal analysis (DAG adjustment, propensity methods, refutation tests): a high-BMI trajectory had the largest robust effect (ATE ≈ +0.54).

## Limitations
Public cohort of 400 patients without eGFR/creatinine; small pre-onset test set; observational causal estimates (positivity violation for diabetes duration); no external validation.

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
├── trajectory_analysis.py   # RF and XGBoost training on trajectory based features
├── causal_analysis.py       # Causal Inference analysis
├── converter_clustering.py  # Clustering patients who transitioned from non-CKD to CKD
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
conda create -n ckd_prediction python=3.14
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
7. Trajectory Analysis
8. Causal Inference Analysis
9. Converter Clustering Analysis
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
- All results are saved to an `results/` directory.

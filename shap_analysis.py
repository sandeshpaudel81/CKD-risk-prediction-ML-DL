# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score
from xgboost                 import XGBClassifier

def create_folder():
    folders = ['plots']
    cwd = os.getcwd()
    for folder in folders:
        folder_name = os.path.join(cwd, folder)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

def compute_shap_values(model, X_test, experiment):
    print(f"\nComputing SHAP values for {experiment} windows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Baseline (expected) prediction: {explainer.expected_value:.4f}")
    return explainer, shap_values

def top_n_shap_features(shap_values, X_test, FEATURE_COLS, n=15):
    shap_df_pre = pd.DataFrame(shap_values, columns=X_test.columns)
    
    # Aggregate: sum absolute SHAP across time steps per base feature
    base_shap = {}
    for feat in FEATURE_COLS:
        cols = [c for c in X_test.columns if c.startswith(feat + '_t')]
        if cols:
            base_shap[feat] = shap_df_pre[cols].abs().sum(axis=1).values
    base_shap_df   = pd.DataFrame(base_shap)
    mean_abs_shap  = base_shap_df.mean().sort_values(ascending=False)
    print(f"\nTop 15 features by mean absolute SHAP:")
    print(mean_abs_shap.head(15).to_string())
    return mean_abs_shap

def main():

    create_folder()

    print("Initializing SHAP analysis...")

    print("\nLoading dataset...")
    df = pd.DataFrame(
        np.load('dataset/clean_data.npy', allow_pickle=True),
        columns=pd.read_csv('dataset/column_names.csv').iloc[:,0].tolist()
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"CKD=1 rows: {df['CKD'].sum()} / {len(df)}")

    print(f"All columns ({len(df.columns)}): {df.columns}")

    # Import trained XGBoost model and test dataset
    print("\nLoading trained XGBoost model and test dataset of all windows...")
    model_xgb_all = joblib.load('rf_xgb_results/model_xgb_all_scale.pkl')
    X_test_all = pd.DataFrame(
                    np.load('rf_xgb_results/X_test_all.npy'),
                    columns=pd.read_csv('rf_xgb_results/feature_names_flat.csv').iloc[:,0].tolist()
                )
    X_train_all = pd.DataFrame(
                    np.load('rf_xgb_results/X_train_all.npy'),
                    columns=X_test_all.columns
                )
    y_test_all = np.load('rf_xgb_results/y_test_all.npy')
    y_train_all = np.load('rf_xgb_results/y_train_all.npy')
    pid_test_all = np.load('rf_xgb_results/pid_test_all.npy')
    print("Loaded saved model and test data.")
    
    print("\nLoading trained XGBoost model and test dataset of pre-onset windows...")
    model_xgb_pre = joblib.load('rf_xgb_results/model_xgb_pre_scale.pkl')
    X_test_pre = pd.DataFrame(
                    np.load('rf_xgb_results/X_test_pre.npy'),
                    columns=pd.read_csv('rf_xgb_results/feature_names_flat.csv').iloc[:,0].tolist()
                )
    X_train_pre = pd.DataFrame(
                    np.load('rf_xgb_results/X_train_pre.npy'),
                    columns=X_test_pre.columns
                )
    y_test_pre = np.load('rf_xgb_results/y_test_pre.npy')
    y_train_pre = np.load('rf_xgb_results/y_train_pre.npy')
    pid_test_pre = np.load('rf_xgb_results/pid_test_pre.npy')
    print("Loaded saved model and test data.")

    print(f"Test dataset for all windows setup: {X_test_all.shape}")
    print(f"Test dataset for pre-onset windows setup: {X_test_pre.shape}")

    # Compute SHAP values for both models
    explainer_all, shap_values_all = compute_shap_values(model_xgb_all, X_test_all, "all windows")
    explainer_pre, shap_values_pre = compute_shap_values(model_xgb_pre, X_test_pre, "pre-onset windows")

    # Top 15 features by mean absolute SHAP for all windows
    WINDOW_SIZE  = 5
    TARGET_COL   = 'CKD'
    FEATURE_COLS = [c for c in df.columns if c not in ['patient_id', TARGET_COL]]
    
    mean_abs_shap_all = top_n_shap_features(shap_values_all, X_test_all, FEATURE_COLS, n=15)
    mean_abs_shap_pre = top_n_shap_features(shap_values_pre, X_test_pre, FEATURE_COLS, n=15)

    # Plotting SHAP summary plots for top features
    fig, axes = plt.subplots(1, 2, figsize=(14, 12))
    for ax, experiment, mean_abs_shap in [
        (axes[0], 'All windows',    mean_abs_shap_all),
        (axes[1], 'Pre-onset only', mean_abs_shap_pre),
    ]:
        top_feat = mean_abs_shap.head(15)
        colors   = ['salmon' if v > 0 else 'steelblue'
                    for v in top_feat.values]
        
        ax.barh(top_feat.index[::-1], top_feat.values[::-1],
                color='steelblue', alpha=0.85)
        ax.set_xlabel('Mean SHAP value (average impact on prediction)')
        ax.set_title(f'Global feature importance — Top 15 features\n'
                    f'XGBoost, {experiment} experiment')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/shap_global_bar.png', dpi=150, bbox_inches='tight')

    # Plotting beeswarm summary plots for top features for all windows
    top15_base  = mean_abs_shap_all.head(15).index.tolist()
    top15_cols  = []
    for feat in top15_base:
        top15_cols += [c for c in X_test_all.columns if c.startswith(feat + '_t')]
    
    # Subset SHAP and X_test for these columns
    shap_top    = shap_values_all[:, [list(X_test_all.columns).index(c) for c in top15_cols]]
    X_test_top  = X_test_all[top15_cols]
    
    # Rename columns to be readable (strip _t0, _t-1 etc for display)
    display_cols = [c.replace('_t0','(t0)').replace('_t-','(t-') for c in top15_cols]
    X_disp       = X_test_top.copy()
    X_disp.columns = display_cols
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_top, X_disp,
                    plot_type='dot',
                    max_display=20,
                    show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_beeswarm.png', dpi=150, bbox_inches='tight')

    # Plotting timestep-level global SHAP values for all windows
    timestep_importance = {}
    for step in range(WINDOW_SIZE):
        suffix = f't-{WINDOW_SIZE-1-step}' if step < WINDOW_SIZE-1 else 't0'
        cols   = [c for c in X_test_all.columns if c.endswith(f'_{suffix}')]
        if cols:
            idx  = [list(X_test_all.columns).index(c) for c in cols]
            timestep_importance[suffix] = np.abs(shap_values_all[:, idx]).mean()
    
    ts_series = pd.Series(timestep_importance)
    print(f"\nSHAP importance by time step:")
    print(ts_series.to_string())
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ts_series.index, ts_series.values,
        color='steelblue', alpha=0.85)
    ax.set_xlabel('Year in 5-year window (t-4 = oldest, t0 = most recent)')
    ax.set_ylabel('Mean |SHAP value| summed across features')
    ax.set_title('Which year in the window contributes most to prediction?\n'
                '(XGBoost SHAP — All windows)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/shap_timestep.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
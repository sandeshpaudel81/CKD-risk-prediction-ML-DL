# Import data processing and visualisation functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle

# Import model development and evaluation functions
from sklearn.metrics import (roc_curve, precision_recall_curve)

import warnings
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`")

# Import custom utility functions
from utils.rf_xgb import (
    build_windows, apply_smote, rf_fit_evaluate, xgb_fit_evaluate,
    scale_splits
)
from utils.common import create_folder, show_results


def main(base_dir):

    create_folder(base_dir, ['results/rf_xgb', 'plots/rf_xgb'])

    PLOT_DIR = base_dir / 'plots/rf_xgb'
    OUT_DIR  = base_dir / 'results/rf_xgb'

    print("Starting model development and evaluation...")
    
    # Import train, validation and test dataframes from file
    print("Loading datasets...")
    train_df = pd.DataFrame(
        np.load(base_dir / 'dataset/train_df.npy', allow_pickle=True),
        columns=pd.read_csv(base_dir / 'dataset/column_names.csv').iloc[:,0].tolist()
    )
    val_df = pd.DataFrame(
        np.load(base_dir / 'dataset/val_df.npy', allow_pickle=True),
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        np.load(base_dir / 'dataset/test_df.npy', allow_pickle=True),
        columns=train_df.columns
    )

    WINDOW_SIZE  = 5
    TARGET_COL   = 'CKD'
    DROP_COLS    = ['patient_id']
    FEATURE_COLS = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # Build sliding windows for all-window experiment
    print("\nBuilding sliding windows for all-window experiment...")
    X_train_all, y_train_all, pid_train_all = build_windows(train_df, FEATURE_COLS, TARGET_COL)
    X_val_all,   y_val_all, pid_val_all   = build_windows(val_df, FEATURE_COLS, TARGET_COL)
    X_test_all,  y_test_all, pid_test_all  = build_windows(test_df, FEATURE_COLS, TARGET_COL)
    print("\n── All windows ──")
    print(f"Train: {X_train_all.shape} CKD=1: {y_train_all.sum():.0f}")
    print(f"Val:   {X_val_all.shape} CKD=1: {y_val_all.sum():.0f}")
    print(f"Test:  {X_test_all.shape} CKD=1: {y_test_all.sum():.0f}")

    # Build sliding windows for pre-onset experiment
    print("\nBuilding sliding windows for pre-onset experiment...")
    X_train_pre, y_train_pre, pid_train_pre = build_windows(train_df, FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    X_val_pre,   y_val_pre, pid_test_pre   = build_windows(val_df,   FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    X_test_pre,  y_test_pre, pid_test_pre  = build_windows(test_df,  FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    print("\n── Pre-onset windows ──")
    print(f"Train: {X_train_pre.shape} CKD=1: {y_train_pre.sum():.0f}")
    print(f"Val: {X_val_pre.shape} CKD=1: {y_val_pre.sum():.0f}")
    print(f"Test: {X_test_pre.shape} CKD=1: {y_test_pre.sum():.0f}")

    # Scale the features for both experiments
    print("\nScaling features...")
    X_train_all_sc, X_val_all_sc, X_test_all_sc, _ = scale_splits(
        X_train_all, X_val_all, X_test_all)
    X_train_pre_sc, X_val_pre_sc, X_test_pre_sc, _ = scale_splits(
        X_train_pre, X_val_pre, X_test_pre)

    # Apply SMOTE to handle class imbalance for both experiments
    print("\nApplying SMOTE to all-windows train...")
    X_train_all_sm, y_train_all_sm = apply_smote(X_train_all_sc, y_train_all)
    print("Applying SMOTE to pre-onset train...")
    X_train_pre_sm, y_train_pre_sm = apply_smote(X_train_pre_sc, y_train_pre, k_neighbors=3)

    print("\n========= Starting model training and evaluation =========")
    print("\n ──--------- Random Forest ------------")

    # RF on all windows
    best_rf_all,rf_all = rf_fit_evaluate(
        X_train_all_sm, y_train_all_sm, 
        X_val_all_sc, y_val_all, 
        X_test_all_sc, y_test_all, experiment='All windows'
    )

    # RF on pre-onset windows
    best_rf_pre, rf_pre = rf_fit_evaluate(
        X_train_pre_sm, y_train_pre_sm, 
        X_val_pre_sc, y_val_pre, 
        X_test_pre_sc, y_test_pre, experiment='Pre-onset only'
    )

    # RF feature importance analysis
    importances_all = pd.Series(best_rf_all.feature_importances_,
                        index=X_train_all_sm.columns).sort_values(ascending=False)
    print("\nTop 20 RF feature importances (All windows):")
    print(importances_all.head(20).to_string())

    importances_pre = pd.Series(best_rf_pre.feature_importances_,
                        index=X_train_pre_sm.columns).sort_values(ascending=False)
    print("\nTop 20 RF feature importances (Pre-onset only):")
    print(importances_pre.head(20).to_string())


    # XGBoost Experiments
    print("\n ──--------- XGBoost ------------")
    print("\nTraining XGBoost on all windows (pos_weight scaled)...")
    data_all_scale = {
        'X_train': X_train_all_sc,
        'y_train': y_train_all,
        'X_val': X_val_all_sc,
        'y_val': y_val_all,
        'X_test': X_test_all_sc,
        'y_test': y_test_all
    }
    xgb_all_scale, res_xgb_all_scale = xgb_fit_evaluate(
        data_all_scale, balance_method='pos_scaled', experiment='All windows'
    )

    print("\nTraining XGBoost on pre-onset windows (pos_weight scaled)...")
    data_pre_scale = {
        'X_train': X_train_pre_sc,
        'y_train': y_train_pre,
        'X_val': X_val_pre_sc,
        'y_val': y_val_pre,
        'X_test': X_test_pre_sc,
        'y_test': y_test_pre
    }
    xgb_pre_scale, res_xgb_pre_scale = xgb_fit_evaluate(
        data_pre_scale, balance_method='pos_scaled', experiment='Pre-onset only'
    )

    print("\nTraining XGBoost on all windows (SMOTE)...")
    data_all_smote = {
        'X_train': X_train_all_sm,
        'y_train': y_train_all_sm,
        'X_val': X_val_all_sc,
        'y_val': y_val_all,
        'X_test': X_test_all_sc,
        'y_test': y_test_all
    }
    xgb_all_smote, res_xgb_all_smote = xgb_fit_evaluate(
        data_all_smote, balance_method='SMOTE', experiment='All windows'
    )

    print("\nTraining XGBoost on pre-onset windows (SMOTE)...")
    data_pre_smote = {
        'X_train': X_train_pre_sm,
        'y_train': y_train_pre_sm,
        'X_val': X_val_pre_sc,
        'y_val': y_val_pre,
        'X_test': X_test_pre_sc,
        'y_test': y_test_pre
    }
    xgb_pre_smote, res_xgb_pre_smote = xgb_fit_evaluate(
        data_pre_smote, balance_method='SMOTE', experiment='Pre-onset only'
    )

    # Feature importance analysis for XGBoost
    xgb_importances = pd.Series(xgb_all_scale.feature_importances_,
                             index=X_train_all_sc.columns).sort_values(ascending=False)
    print("\nTop 20 XGBoost feature importances:")
    print(xgb_importances.head(20).to_string())

    print("\n========= Summary of Results =========")
    show_results('Random Forest', [rf_all, rf_pre])
    show_results('XGBoost (pos_scale balanced)', [res_xgb_all_scale, res_xgb_pre_scale])
    show_results('XGBoost (SMOTE balanced)', [res_xgb_all_smote, res_xgb_pre_smote])

    print("\nResult visualization and saving results to files ...")

    # Plotting PR curves for all models and experiments
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, exp_label, y_test_exp, res_list in [
        (axes[0], 'All windows',    y_test_all, [
            (rf_all,              'Random Forest'),
            (res_xgb_all_scale,   'XGBoost (pos_scale)'),
            (res_xgb_all_smote,   'XGBoost (SMOTE)'),
        ]),
        (axes[1], 'Pre-onset only', y_test_pre, [
            (rf_pre,              'Random Forest'),
            (res_xgb_pre_scale,   'XGBoost (pos_scale)'),
            (res_xgb_pre_smote,   'XGBoost (SMOTE)'),
        ]),
    ]:
        for r, label in res_list:
            prec, rec, _ = precision_recall_curve(y_test_exp, r['y_prob'])
            ax.plot(rec, prec, lw=2,
                    label=f"{label} (PR-AUC={r['pr_auc']:.3f})")
        baseline = y_test_exp.mean()
        ax.axhline(baseline, color='k', linestyle='--', lw=1,
                label=f'Baseline ({baseline:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — {exp_label}')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/tree_pr_curves.png', dpi=150, bbox_inches='tight')
    
    # Plotting ROC curves for all models and experiments
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, exp_label, y_test_exp, res_list in [
        (axes[0], 'All windows',    y_test_all, [
            (rf_all,            'Random Forest'),
            (res_xgb_all_scale, 'XGBoost (pos_scale)'),
            (res_xgb_all_smote, 'XGBoost (SMOTE)'),
        ]),
        (axes[1], 'Pre-onset only', y_test_pre, [
            (rf_pre,            'Random Forest'),
            (res_xgb_pre_scale, 'XGBoost (pos_scale)'),
            (res_xgb_pre_smote, 'XGBoost (SMOTE)'),
        ]),
    ]:
        for r, label in res_list:
            fpr, tpr, _ = roc_curve(y_test_exp, r['y_prob'])
            ax.plot(fpr, tpr, lw=2,
                    label=f"{label} (AUC={r['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {exp_label}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/tree_roc_curves.png', dpi=150, bbox_inches='tight')

    # Plotting feature importances
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, title, imp in [
        (axes[0], 'Random Forest — Top 20 Features', importances_all.head(20)),
        (axes[1], 'XGBoost — Top 20 Features',       xgb_importances.head(20)),
    ]:
        imp.sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')

    # Saving models, test data, and results to files
    joblib.dump(best_rf_all, f'{OUT_DIR}/model_rf_all.pkl')
    joblib.dump(best_rf_pre, f'{OUT_DIR}/model_rf_pre.pkl')
    joblib.dump(xgb_all_scale, f'{OUT_DIR}/model_xgb_all_scale.pkl')
    joblib.dump(xgb_pre_scale, f'{OUT_DIR}/model_xgb_pre_scale.pkl')
    joblib.dump(xgb_all_smote, f'{OUT_DIR}/model_xgb_all_smote.pkl')
    joblib.dump(xgb_pre_smote, f'{OUT_DIR}/model_xgb_pre_smote.pkl')

    # Save test data + predictions
    np.save(f'{OUT_DIR}/X_test_all.npy',    X_test_all_sc.values)
    np.save(f'{OUT_DIR}/y_test_all.npy',    y_test_all)
    np.save(f'{OUT_DIR}/X_train_all.npy',   X_train_all_sc.values)
    np.save(f'{OUT_DIR}/y_train_all.npy',   y_train_all)
    np.save(f'{OUT_DIR}/X_test_pre.npy',    X_test_pre_sc.values)
    np.save(f'{OUT_DIR}/y_test_pre.npy',    y_test_pre)
    np.save(f'{OUT_DIR}/X_train_pre.npy',   X_train_pre_sc.values)
    np.save(f'{OUT_DIR}/y_train_pre.npy',   y_train_pre)

    # Save test patients id
    np.save(f'{OUT_DIR}/pid_test_all.npy',  pid_test_all)
    np.save(f'{OUT_DIR}/pid_test_pre.npy',  pid_test_pre)

    # Save column names (needed for SHAP feature names)
    pd.Series(X_train_all_sc.columns).to_csv(f'{OUT_DIR}/feature_names_flat.csv', index=False)

    # Save results dict
    with open(f'{OUT_DIR}/results_rf_xgb.pkl', 'wb') as f:
        pickle.dump({'rf_all': rf_all, 'rf_pre': rf_pre,
                    'res_xgb_all_scale': res_xgb_all_scale, 'res_xgb_pre_scale': res_xgb_pre_scale,
                    'res_xgb_all_smote': res_xgb_all_smote, 'res_xgb_pre_smote': res_xgb_pre_smote}, f)
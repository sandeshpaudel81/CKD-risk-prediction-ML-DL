# Import data processing and visualisation functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle

# Import model development and evaluation functions
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                                        roc_auc_score, average_precision_score,
                                        roc_curve, precision_recall_curve, f1_score, recall_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def create_folder():
    folders = ['plots', 'rf_xgb_results']
    cwd = os.getcwd()
    for folder in folders:
        folder_name = os.path.join(cwd, folder)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

def build_windows(data_df, feature_cols, target_col, window_size=5, pre_onset_only=False):
    X_rows, y_rows, pid_list = [], [], []
    for pid, group in data_df.groupby('patient_id'):
        group    = group.sort_values('diabetic_year')
        features = group[feature_cols].values  # shape (n, feat+1)
        labels   = group[target_col].values
        n        = len(features)
        for i in range(n - window_size):
            window_feats  = features[i : i + window_size]   # (window, n_features)
            window_labels = labels[i : i + window_size]
            target        = labels[i + window_size]
            if pre_onset_only and window_labels.sum() > 0:
                continue
            # Flatten and append sequences to the list 
            X_rows.append(window_feats.flatten())
            y_rows.append(target)
            pid_list.append(pid)
    # Build column names for interpretability
    col_names = [
        f"{feat}_t{-(window_size - 1 - step)}" if (window_size - 1 - step) != 0
        else f"{feat}_t0"
        for step in range(window_size)
        for feat in feature_cols
    ]
    # t-4,t-3,t-2,t-1,t0 for first,second,third,forth, and fifth year respectively
    X = pd.DataFrame(X_rows, columns=col_names)
    y = np.array(y_rows, dtype=np.float32)
    pidList = np.array(pid_list, dtype=np.int32)
    return X, y, pidList

def scale_splits(X_train, X_val, X_test):
    scaler   = StandardScaler()
    X_train_sc  = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_sc, X_val_sc, X_test_sc, scaler

def apply_smote(X_train, y_train, k_neighbors=5):
    k = min(k_neighbors, int(y_train.sum()) - 1)  # safe k
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f" Before SMOTE - CKD=0: {(y_train==0).sum()}  CKD=1: {(y_train==1).sum()}")
    print(f"  After SMOTE — CKD=0: {(y_res==0).sum()}  CKD=1: {(y_res==1).sum()}")
    return X_res, y_res

# Function to find best threshold 
# Two methods: 
# 1. Youden's J statistic method with ROC Curve
# 2. Precision-Recall Curve Optimization
def find_best_threshold(y_true, y_prob, method='youden'):
    if method == 'youden':
        print(f"\nUsing Youden's J statistics on val..")
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_thresh = thresholds[np.argmax(j_scores)]
    else:
        print(f"\nUsing PR curve optimization on val..")
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresh = thresholds[np.argmax(f1[:-1])]
    return float(best_thresh)

# Function to evaluate the model using validation and test sets

def evaluate(model, X_test, y_test, X_val, y_val, model_name, experiment):
 
    print(f"\n{model_name}  |  {experiment}")
    
    y_val_prob  = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
 
    # Tune threshold on validation set
    thresh = find_best_threshold(y_val, y_val_prob, method='youden')
    print(f"\nOptimal threshold (Youden's J on val): {thresh:.3f}")
 
    y_pred = (y_test_prob >= thresh).astype(int)
 
    print(f"\n{classification_report(y_test, y_pred, target_names=['No CKD','CKD'])}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_test_prob):.4f}")
    print(f"  PR-AUC   : {average_precision_score(y_test, y_test_prob):.4f}")
    print(f"\n  Confusion matrix:")
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f"  TN={conf_mat[0,0]}  FP={conf_mat[0,1]}")
    print(f"  FN={conf_mat[1,0]}  TP={conf_mat[1,1]}")
 
    return {
        'model'      : model_name,
        'experiment' : experiment,
        'threshold'  : thresh,
        'roc_auc'    : roc_auc_score(y_test, y_test_prob),
        'pr_auc'     : average_precision_score(y_test, y_test_prob),
        'recall_ckd' : recall_score(y_test, y_pred),
        'f1_ckd'     : f1_score(y_test, y_pred),
        'y_prob'     : y_test_prob,
        'y_pred'     : y_pred,
        'cm'         : conf_mat,
    }

def rf_fit_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, experiment):
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=30,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    rf_search = rf_search.fit(X_train, y_train)
    print(f"\nBest RF params: {rf_search.best_params_}")
    print(f"Best CV AUC   : {rf_search.best_score_:.4f}")

    best_rf = rf_search.best_estimator_
    rf_results = evaluate(best_rf,
                  X_test, y_test,
                  X_val,  y_val,
                  model_name='Random Forest', experiment=experiment)
    return best_rf, rf_results

def xgb_fit_evaluate(data, balance_method, experiment):
    print(f"\n XGBoost model training on {balance_method} data for {experiment} experiment..")
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scale = ((y_train == 0).sum()) / (y_train.sum())
    if (balance_method == 'pos_scaled'):
        print(f"\nscale_pos_weight: {scale:.2f}")
    xgb_param_grid = {
        'n_estimators'      : [200, 300, 500],
        'max_depth'         : [3, 4, 6, 8],
        'learning_rate'     : [0.01, 0.05, 0.1, 0.2],
        'subsample'         : [0.6, 0.8, 1.0],
        'colsample_bytree'  : [0.6, 0.8, 1.0],
        'min_child_weight'  : [1, 3, 5],
        'gamma'             : [0, 0.1, 0.3],
        'reg_alpha'         : [0, 0.1, 1.0],
        'reg_lambda'        : [1.0, 5.0, 10.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        param_distributions=xgb_param_grid,
        n_iter=30,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    xgb_search.fit(X_train, y_train)
    print(f"\nBest XGB params: {xgb_search.best_params_}")
    print(f"Best CV AUC    : {xgb_search.best_score_:.4f}")
    best_xgb = xgb_search.best_estimator_
    xgb_res = evaluate(best_xgb,
                   X_test, y_test,
                   X_val,  y_val,
                   model_name='XGBoost', experiment=experiment)
    return best_xgb, xgb_res

def show_results(model, results):
    print(f"\n{model} Results:")
    summary = pd.DataFrame([{
        'Model'       : r['model'],
        'Experiment'  : r['experiment'],
        'ROC-AUC'     : round(r['roc_auc'], 4),
        'PR-AUC'      : round(r['pr_auc'],  4),
        'F1 (CKD)'    : round(r['f1_ckd'],  4),
        'Recall (CKD)': round(r['recall_ckd'], 4),
        'Threshold'   : round(r['threshold'], 3)
    } for r in results])
    print(summary.to_string(index=False))

def main():
    create_folder()
    print("Starting model development and evaluation...")
    
    # Import train, validation and test dataframes from file
    print("Loading datasets...")
    train_df = pd.DataFrame(
        np.load('dataset/train_df.npy', allow_pickle=True),
        columns=pd.read_csv('dataset/column_names.csv').iloc[:,0].tolist()
    )
    val_df = pd.DataFrame(
        np.load('dataset/val_df.npy', allow_pickle=True),
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        np.load('dataset/test_df.npy', allow_pickle=True),
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
    
    # Parameter search grid for Random Forest
    rf_param_grid = {
        'n_estimators'      : [200, 300, 500],
        'max_depth'         : [None, 10, 20, 30],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf'  : [1, 2, 4],
        'max_features'      : ['sqrt', 'log2'],
        'class_weight'      : ['balanced', {0:1, 1:3}, {0:1, 1:5}]
    }

    # RF on all windows
    best_rf_all,rf_all = rf_fit_evaluate(
        X_train_all_sm, y_train_all_sm, 
        X_val_all_sc, y_val_all, 
        X_test_all_sc, y_test_all, rf_param_grid, experiment='All windows'
    )

    # RF on pre-onset windows
    best_rf_pre, rf_pre = rf_fit_evaluate(
        X_train_pre_sm, y_train_pre_sm, 
        X_val_pre_sc, y_val_pre, 
        X_test_pre_sc, y_test_pre, rf_param_grid, experiment='Pre-onset only'
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
    plt.savefig('plots/tree_pr_curves.png', dpi=150, bbox_inches='tight')
    
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
    plt.savefig('plots/tree_roc_curves.png', dpi=150, bbox_inches='tight')

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
    plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')

    # Saving models, test data, and results to files
    joblib.dump(best_rf_all, 'rf_xgb_results/model_rf_all.pkl')
    joblib.dump(best_rf_pre, 'rf_xgb_results/model_rf_pre.pkl')
    joblib.dump(xgb_all_scale, 'rf_xgb_results/model_xgb_all_scale.pkl')
    joblib.dump(xgb_pre_scale, 'rf_xgb_results/model_xgb_pre_scale.pkl')
    joblib.dump(xgb_all_smote, 'rf_xgb_results/model_xgb_all_smote.pkl')
    joblib.dump(xgb_pre_smote, 'rf_xgb_results/model_xgb_pre_smote.pkl')

    # Save test data + predictions
    np.save('rf_xgb_results/X_test_all.npy',    X_test_all_sc.values)
    np.save('rf_xgb_results/y_test_all.npy',    y_test_all)
    np.save('rf_xgb_results/X_train_all.npy',   X_train_all_sc.values)
    np.save('rf_xgb_results/y_train_all.npy',   y_train_all)
    np.save('rf_xgb_results/X_test_pre.npy',    X_test_pre_sc.values)
    np.save('rf_xgb_results/y_test_pre.npy',    y_test_pre)
    np.save('rf_xgb_results/X_train_pre.npy',   X_train_pre_sc.values)
    np.save('rf_xgb_results/y_train_pre.npy',   y_train_pre)

    # Save test patients id
    np.save('rf_xgb_results/pid_test_all.npy',  pid_test_all)
    np.save('rf_xgb_results/pid_test_pre.npy',  pid_test_pre)

    # Save column names (needed for SHAP feature names)
    pd.Series(X_train_all_sc.columns).to_csv('rf_xgb_results/feature_names_flat.csv', index=False)

    # Save results dict
    with open('rf_xgb_results/results_rf_xgb.pkl', 'wb') as f:
        pickle.dump({'rf_all': rf_all, 'rf_pre': rf_pre,
                    'res_xgb_all_scale': res_xgb_all_scale, 'res_xgb_pre_scale': res_xgb_pre_scale,
                    'res_xgb_all_smote': res_xgb_all_smote, 'res_xgb_pre_smote': res_xgb_pre_smote}, f)

if __name__ == "__main__":
    main()
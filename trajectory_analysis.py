import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
import shap

from scipy.stats import linregress
from sklearn.metrics import (roc_curve, precision_recall_curve)

from utils.rf_xgb import (
    xgb_fit_evaluate, rf_fit_evaluate, apply_smote, 
    scale_splits
)

from utils.common import create_folder, show_results


# ── Trajectory feature config ─────────────────────────────────────────────────

SLOPE_FEATS = [
    'weight_in_kg', 'BMI_in_kg_per_m2', 'comorbidity_score', 'lifestyle_score'
]
VAR_FEATS = [
    'weight_in_kg', 'BMI_in_kg_per_m2', 'comorbidity_score', 'lifestyle_score'
]
ACCEL_FEATS = [
    'weight_in_kg', 'BMI_in_kg_per_m2'
]
RECENCY_FEATS = [
    'weight_in_kg', 'BMI_in_kg_per_m2', 'comorbidity_score'
]
BINARY_SUM_FEATS = [
    'takes_insulin', 'urinary_infection', 'has_heart_disease', 'takes_pain_killer'
]
BINARY_FIRST_FEATS = [
    'takes_insulin', 'urinary_infection', 'has_heart_disease', 'takes_pain_killer'
]

WINDOW_SIZE = 5
EARLY_DECAY  = np.array([0.4, 0.25, 0.15, 0.1, 0.1])   # weights t-4..t0
RECENCY_DECAY = np.array([0.1, 0.1, 0.15, 0.25, 0.4])   # weights t-4..t0


# ── Window builder (same as rf_xgboost.py) ───────────────────────────────────

def build_windows(data_df, feature_cols, target_col, window_size=5, pre_onset_only=False):
    X_rows, y_rows, pid_list = [], [], []
    for pid, group in data_df.groupby('patient_id'):
        group    = group.sort_values('diabetic_year')
        features = group[feature_cols].values
        labels   = group[target_col].values
        n        = len(features)
        for i in range(n - window_size):
            window_feats  = features[i: i + window_size]
            window_labels = labels[i: i + window_size]
            target        = labels[i + window_size]
            if pre_onset_only and window_labels.sum() > 0:
                continue
            X_rows.append(window_feats.flatten())
            y_rows.append(target)
            pid_list.append(pid)
    col_names = [
        f"{feat}_t{-(window_size - 1 - step)}" if (window_size - 1 - step) != 0
        else f"{feat}_t0"
        for step in range(window_size)
        for feat in feature_cols
    ]
    X = pd.DataFrame(X_rows, columns=col_names)
    y = np.array(y_rows, dtype=np.float32)
    pids = np.array(pid_list, dtype=np.int32)
    return X, y, pids


# ── Trajectory feature extraction ─────────────────────────────────────────────

def _get_window_vals(X_flat, feat, window_size=5):
    cols = [
        f"{feat}_t{-(window_size - 1 - step)}" if (window_size - 1 - step) != 0
        else f"{feat}_t0"
        for step in range(window_size)
    ]
    return X_flat[cols].values


def add_trajectory_features(X_flat, window_size=5):
    traj = {}

    for feat in SLOPE_FEATS:
        vals = _get_window_vals(X_flat, feat, window_size)
        x    = np.arange(window_size)
        slopes = np.apply_along_axis(
            lambda v: linregress(x, v).slope, axis=1, arr=vals
        )
        traj[f"{feat}_slope"] = slopes

    for feat in VAR_FEATS:
        vals = _get_window_vals(X_flat, feat, window_size)
        traj[f"{feat}_std"] = vals.std(axis=1)

    for feat in ACCEL_FEATS:
        vals  = _get_window_vals(X_flat, feat, window_size)
        half  = window_size // 2
        early = vals[:, :half + 1]
        late  = vals[:, half:]
        x_e   = np.arange(early.shape[1])
        x_l   = np.arange(late.shape[1])
        slope_early = np.apply_along_axis(lambda v: linregress(x_e, v).slope, 1, early)
        slope_late  = np.apply_along_axis(lambda v: linregress(x_l, v).slope, 1, late)
        traj[f"{feat}_accel"] = slope_late - slope_early

    for feat in RECENCY_FEATS:
        vals = _get_window_vals(X_flat, feat, window_size)
        traj[f"{feat}_recency_wmean"] = (vals * RECENCY_DECAY).sum(axis=1)
        traj[f"{feat}_early_wmean"]   = (vals * EARLY_DECAY).sum(axis=1)

    for feat in BINARY_SUM_FEATS:
        vals = _get_window_vals(X_flat, feat, window_size)
        traj[f"{feat}_window_sum"] = vals.sum(axis=1)

    for feat in BINARY_FIRST_FEATS:
        vals = _get_window_vals(X_flat, feat, window_size)
        first_occ = np.where(vals > 0, np.arange(window_size), window_size)
        traj[f"{feat}_first_occur"] = first_occ.min(axis=1).astype(float)

    return pd.DataFrame(traj, index=X_flat.index)


def build_augmented(X_flat, window_size=5):
    traj_df = add_trajectory_features(X_flat, window_size)
    return pd.concat([X_flat.reset_index(drop=True),
                      traj_df.reset_index(drop=True)], axis=1)


def build_replace(X_flat, window_size=5):
    return add_trajectory_features(X_flat, window_size).reset_index(drop=True)


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def run_shap(model, X_train, X_test, label, out_dir, top_n=20):
    print(f"\nComputing SHAP values for {label}...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    mean_shap  = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)

    print(f"  Top {top_n} SHAP features ({label}):")
    print(mean_shap.head(top_n).to_string())

    # Beeswarm summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test, max_display=top_n, show=False)
    plt.title(f'SHAP Summary — {label}')
    plt.tight_layout()
    fname = label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    plt.savefig(f'{out_dir}/shap_summary_{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Bar plot of mean |SHAP|
    plt.figure(figsize=(10, 6))
    mean_shap.head(top_n).sort_values().plot(kind='barh', color='steelblue')
    plt.title(f'Mean |SHAP| — {label}')
    plt.xlabel('Mean |SHAP value|')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/shap_bar_{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # SHAP interaction values — timestep aggregation
    traj_names = [c for c in X_test.columns if any(
        c.endswith(s) for s in ['_slope', '_std', '_accel', '_wmean', '_window_sum', '_first_occur']
    )]
    raw_names  = [c for c in X_test.columns if c not in traj_names]

    traj_shap = np.abs(shap_vals[:, [X_test.columns.get_loc(c) for c in traj_names]]).mean()
    raw_shap  = np.abs(shap_vals[:, [X_test.columns.get_loc(c) for c in raw_names]]).mean()
    print(f"\n  Mean |SHAP| — raw timestep features   : {raw_shap:.4f}")
    print(f"  Mean |SHAP| — trajectory features     : {traj_shap:.4f}")

    return mean_shap, shap_vals


def run_shap_timestep(model, X_test, label, out_dir):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    timesteps = ['t-4', 't-3', 't-2', 't-1', 't0']
    ts_importance = {}
    for ts in timesteps:
        cols = [c for c in X_test.columns if c.endswith(ts)]
        if cols:
            idx = [X_test.columns.get_loc(c) for c in cols]
            ts_importance[ts] = np.abs(shap_vals[:, idx]).mean()
    print(f"\n  SHAP by timestep ({label}):")
    for k, v in ts_importance.items():
        print(f"    {k}  {v:.6f}")

    plt.figure(figsize=(7, 4))
    plt.bar(ts_importance.keys(), ts_importance.values(), color='steelblue')
    plt.title(f'SHAP Importance by Timestep — {label}')
    plt.ylabel('Mean |SHAP value|')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    plt.savefig(f'{out_dir}/shap_timestep_{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_roc_pr(results_all, results_pre, y_test_all, y_test_pre, suffix, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, exp_label, y_test, res_list in [
        (axes[0], 'All Windows',    y_test_all, results_all),
        (axes[1], 'Pre-onset Only', y_test_pre, results_pre),
    ]:
        for r, label in res_list:
            fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={r['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {exp_label}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/roc_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, exp_label, y_test, res_list in [
        (axes[0], 'All Windows',    y_test_all, results_all),
        (axes[1], 'Pre-onset Only', y_test_pre, results_pre),
    ]:
        for r, label in res_list:
            prec, rec, _ = precision_recall_curve(y_test, r['y_prob'])
            ax.plot(rec, prec, lw=2, label=f"{label} (PR-AUC={r['pr_auc']:.3f})")
        ax.axhline(y_test.mean(), color='k', linestyle='--', lw=1,
                   label=f'Baseline ({y_test.mean():.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — {exp_label}')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/pr_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances_dict, top_n, suffix, out_dir):
    n = len(importances_dict)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    if n == 1:
        axes = [axes]
    for ax, (title, imp) in zip(axes, importances_dict.items()):
        imp.head(top_n).sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/feature_importance_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory_vs_raw_shap(mean_shap_aug, out_dir, top_n=15):
    traj_names = [c for c in mean_shap_aug.index if any(
        c.endswith(s) for s in ['_slope', '_std', '_accel', '_wmean', '_window_sum', '_first_occur']
    )]
    raw_names = [c for c in mean_shap_aug.index if c not in traj_names]

    traj_top = mean_shap_aug[traj_names].head(top_n)
    raw_top  = mean_shap_aug[raw_names].head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    raw_top.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Top Raw Timestep Features by Mean |SHAP|')
    axes[0].set_xlabel('Mean |SHAP value|')
    axes[0].grid(axis='x', alpha=0.3)

    traj_top.sort_values().plot(kind='barh', ax=axes[1], color='darkorange')
    axes[1].set_title('Top Trajectory Features by Mean |SHAP|')
    axes[1].set_xlabel('Mean |SHAP value|')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/shap_raw_vs_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(baseline_results, traj_aug_results, traj_rep_results, out_dir):
    metrics = ['roc_auc', 'pr_auc', 'f1_ckd', 'recall_ckd']
    labels  = ['ROC-AUC', 'PR-AUC', 'F1 (CKD)', 'Recall (CKD)']
 
    # explicit pairing: (display_label, baseline_r, aug_r, rep_r)
    pairs = [
        ('Random Forest — All Windows',
         next((r for r in baseline_results if 'Random Forest' in r['model'] and 'All' in r['experiment']), None),
         next((r for r in traj_aug_results  if 'Random Forest' in r['model'] and 'All' in r['experiment']), None),
         next((r for r in traj_rep_results  if 'Random Forest' in r['model'] and 'All' in r['experiment']), None)),
        ('Random Forest — Pre-onset Only',
         next((r for r in baseline_results if 'Random Forest' in r['model'] and 'Pre' in r['experiment']), None),
         next((r for r in traj_aug_results  if 'Random Forest' in r['model'] and 'Pre' in r['experiment']), None),
         next((r for r in traj_rep_results  if 'Random Forest' in r['model'] and 'Pre' in r['experiment']), None)),
        ('XGBoost — All Windows',
         next((r for r in baseline_results if 'XGBoost' in r['model'] and 'All' in r['experiment']), None),
         next((r for r in traj_aug_results  if 'XGBoost' in r['model'] and 'All' in r['experiment']), None),
         next((r for r in traj_rep_results  if 'XGBoost' in r['model'] and 'All' in r['experiment']), None)),
        ('XGBoost — Pre-onset Only',
         next((r for r in baseline_results if 'XGBoost' in r['model'] and 'Pre' in r['experiment']), None),
         next((r for r in traj_aug_results  if 'XGBoost' in r['model'] and 'Pre' in r['experiment']), None),
         next((r for r in traj_rep_results  if 'XGBoost' in r['model'] and 'Pre' in r['experiment']), None)),
    ]
 
    x     = np.arange(len(metrics))
    width = 0.2
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
 
    for ax, (title, b, a, rp) in zip(axes, pairs):
        b_vals  = [b[m]  for m in metrics] if b  else [0]*4
        a_vals  = [a[m]  for m in metrics] if a  else [0]*4
        rp_vals = [rp[m] for m in metrics] if rp else [0]*4
        ax.bar(x - width, b_vals,  width, label='Original',  color='steelblue')
        ax.bar(x,         a_vals,  width, label='Augmented', color='darkorange')
        ax.bar(x + width, rp_vals, width, label='Replace',   color='seagreen')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(f'{out_dir}/comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_dir):
    create_folder(base_dir, ['plots/trajectory', 'results/trajectory'])
    OUT = f'{base_dir}/plots/trajectory'
    RESULTS_OUT = f'{base_dir}/results/trajectory'

    print("Loading datasets...")
    train_df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/train_df.npy', allow_pickle=True),
        columns=pd.read_csv(f'{base_dir}/dataset/column_names.csv').iloc[:, 0].tolist()
    )
    val_df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/val_df.npy', allow_pickle=True),
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/test_df.npy', allow_pickle=True),
        columns=train_df.columns
    )

    TARGET_COL   = 'CKD'
    DROP_COLS    = ['patient_id']
    FEATURE_COLS = [c for c in train_df.columns if c not in DROP_COLS]
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # ── Build raw flat windows ────────────────────────────────────────────────
    print("\nBuilding sliding windows...")
    X_tr_all, y_tr_all, _ = build_windows(train_df, FEATURE_COLS, TARGET_COL)
    X_va_all, y_va_all, _ = build_windows(val_df,   FEATURE_COLS, TARGET_COL)
    X_te_all, y_te_all, _ = build_windows(test_df,  FEATURE_COLS, TARGET_COL)

    X_tr_pre, y_tr_pre, _ = build_windows(train_df, FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    X_va_pre, y_va_pre, _ = build_windows(val_df,   FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    X_te_pre, y_te_pre, _ = build_windows(test_df,  FEATURE_COLS, TARGET_COL, pre_onset_only=True)

    print(f"  All windows  — Train {X_tr_all.shape}  Val {X_va_all.shape}  Test {X_te_all.shape}")
    print(f"  Pre-onset    — Train {X_tr_pre.shape}  Val {X_va_pre.shape}  Test {X_te_pre.shape}")

    # ── Build augmented and replace feature sets ──────────────────────────────
    print("\nBuilding trajectory feature sets (augmented + replace)...")

    X_tr_all_aug = build_augmented(X_tr_all)
    X_va_all_aug = build_augmented(X_va_all)
    X_te_all_aug = build_augmented(X_te_all)

    X_tr_pre_aug = build_augmented(X_tr_pre)
    X_va_pre_aug = build_augmented(X_va_pre)
    X_te_pre_aug = build_augmented(X_te_pre)

    X_tr_all_rep = build_replace(X_tr_all)
    X_va_all_rep = build_replace(X_va_all)
    X_te_all_rep = build_replace(X_te_all)

    X_tr_pre_rep = build_replace(X_tr_pre)
    X_va_pre_rep = build_replace(X_va_pre)
    X_te_pre_rep = build_replace(X_te_pre)

    n_traj = X_tr_all_aug.shape[1] - X_tr_all.shape[1]
    print(f"  Raw features       : {X_tr_all.shape[1]}")
    print(f"  Trajectory features: {n_traj}")
    print(f"  Augmented total    : {X_tr_all_aug.shape[1]}")
    print(f"  Replace total      : {X_tr_all_rep.shape[1]}")
    print(f"  Trajectory feature names: {list(X_tr_all_aug.columns[X_tr_all.shape[1]:])}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    print("\nScaling features...")
    X_tr_all_aug_sc, X_va_all_aug_sc, X_te_all_aug_sc, _ = scale_splits(X_tr_all_aug, X_va_all_aug, X_te_all_aug)
    X_tr_pre_aug_sc, X_va_pre_aug_sc, X_te_pre_aug_sc, _ = scale_splits(X_tr_pre_aug, X_va_pre_aug, X_te_pre_aug)
    X_tr_all_rep_sc, X_va_all_rep_sc, X_te_all_rep_sc, _ = scale_splits(X_tr_all_rep, X_va_all_rep, X_te_all_rep)
    X_tr_pre_rep_sc, X_va_pre_rep_sc, X_te_pre_rep_sc, _ = scale_splits(X_tr_pre_rep, X_va_pre_rep, X_te_pre_rep)

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    print("\nApplying SMOTE...")
    X_tr_all_aug_sm, y_tr_all_aug_sm = apply_smote(X_tr_all_aug_sc, y_tr_all)
    X_tr_pre_aug_sm, y_tr_pre_aug_sm = apply_smote(X_tr_pre_aug_sc, y_tr_pre, k_neighbors=3)
    X_tr_all_rep_sm, y_tr_all_rep_sm = apply_smote(X_tr_all_rep_sc, y_tr_all)
    X_tr_pre_rep_sm, y_tr_pre_rep_sm = apply_smote(X_tr_pre_rep_sc, y_tr_pre, k_neighbors=3)

    # ═════════════════════════════════════════════════════════════════════════
    # RANDOM FOREST — Augmented
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n══════ Random Forest — Augmented ══════")

    print("\nAll Windows...")
    rf_aug_all, res_rf_aug_all = rf_fit_evaluate(
        X_tr_all_aug_sm, y_tr_all_aug_sm,
        X_va_all_aug_sc, y_va_all,
        X_te_all_aug_sc, y_te_all,
        experiment='All Windows (Augmented)'
    )

    print("\nPre-onset Only...")
    rf_aug_pre, res_rf_aug_pre = rf_fit_evaluate(
        X_tr_pre_aug_sm, y_tr_pre_aug_sm,
        X_va_pre_aug_sc, y_va_pre,
        X_te_pre_aug_sc, y_te_pre,
        experiment='Pre-onset Only (Augmented)'
    )

    imp_rf_aug_all = pd.Series(
        rf_aug_all.feature_importances_, index=X_tr_all_aug_sc.columns
    ).sort_values(ascending=False)
    imp_rf_aug_pre = pd.Series(
        rf_aug_pre.feature_importances_, index=X_tr_pre_aug_sc.columns
    ).sort_values(ascending=False)

    print("\nTop 20 RF (Augmented) — All Windows:")
    print(imp_rf_aug_all.head(20).to_string())
    print("\nTop 20 RF (Augmented) — Pre-onset Only:")
    print(imp_rf_aug_pre.head(20).to_string())

    # ═════════════════════════════════════════════════════════════════════════
    # RANDOM FOREST — Replace
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n══════ Random Forest — Replace ══════")

    print("\nAll Windows...")
    rf_rep_all, res_rf_rep_all = rf_fit_evaluate(
        X_tr_all_rep_sm, y_tr_all_rep_sm,
        X_va_all_rep_sc, y_va_all,
        X_te_all_rep_sc, y_te_all,
        experiment='All Windows (Replace)'
    )

    print("\nPre-onset Only...")
    rf_rep_pre, res_rf_rep_pre = rf_fit_evaluate(
        X_tr_pre_rep_sm, y_tr_pre_rep_sm,
        X_va_pre_rep_sc, y_va_pre,
        X_te_pre_rep_sc, y_te_pre,
        experiment='Pre-onset Only (Replace)'
    )

    imp_rf_rep_all = pd.Series(
        rf_rep_all.feature_importances_, index=X_tr_all_rep_sc.columns
    ).sort_values(ascending=False)
    imp_rf_rep_pre = pd.Series(
        rf_rep_pre.feature_importances_, index=X_tr_pre_rep_sc.columns
    ).sort_values(ascending=False)

    print("\nTop 20 RF (Replace) — All Windows:")
    print(imp_rf_rep_all.head(20).to_string())
    print("\nTop 20 RF (Replace) — Pre-onset Only:")
    print(imp_rf_rep_pre.head(20).to_string())

    # ═════════════════════════════════════════════════════════════════════════
    # XGBOOST — Augmented
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n══════ XGBoost — Augmented ══════")

    xgb_aug_all_pw, res_xgb_aug_all_pw = xgb_fit_evaluate(
        data = {
            'X_train': X_tr_all_aug_sc, 'y_train': y_tr_all,
            'X_val': X_va_all_aug_sc, 'y_val': y_va_all,
            'X_test': X_te_all_aug_sc, 'y_test': y_te_all
        },
        balance_method='pos_weight', experiment='All Windows (Augmented)'
    )
    xgb_aug_pre_pw, res_xgb_aug_pre_pw = xgb_fit_evaluate(
        data = {
            'X_train': X_tr_pre_aug_sc, 'y_train': y_tr_pre,
            'X_val': X_va_pre_aug_sc, 'y_val': y_va_pre,
            'X_test': X_te_pre_aug_sc, 'y_test': y_te_pre
        },
        balance_method='pos_weight', experiment='Pre-onset Only (Augmented)'
    )

    # ═════════════════════════════════════════════════════════════════════════
    # XGBOOST — Replace
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n══════ XGBoost — Replace ══════")

    xgb_rep_all_pw, res_xgb_rep_all_pw = xgb_fit_evaluate(
        data = {
        'X_train': X_tr_all_rep_sc, 'y_train': y_tr_all,
        'X_val': X_va_all_rep_sc, 'y_val': y_va_all,
        'X_test': X_te_all_rep_sc, 'y_test': y_te_all
        },
        balance_method='pos_weight', experiment='All Windows (Replace)'
    )
    xgb_rep_pre_pw, res_xgb_rep_pre_pw = xgb_fit_evaluate(
        data = {
        'X_train': X_tr_pre_rep_sc, 'y_train': y_tr_pre,
        'X_val': X_va_pre_rep_sc, 'y_val': y_va_pre,
        'X_test': X_te_pre_rep_sc, 'y_test': y_te_pre
        },
        balance_method='pos_weight', experiment='Pre-onset Only (Replace)'
    )

    # XGBoost feature importances
    imp_xgb_aug_all = pd.Series(
        xgb_aug_all_pw.feature_importances_, index=X_tr_all_aug_sc.columns
    ).sort_values(ascending=False)
    imp_xgb_aug_pre = pd.Series(
        xgb_aug_pre_pw.feature_importances_, index=X_tr_pre_aug_sc.columns
    ).sort_values(ascending=False)
    imp_xgb_rep_all = pd.Series(
        xgb_rep_all_pw.feature_importances_, index=X_tr_all_rep_sc.columns
    ).sort_values(ascending=False)
    imp_xgb_rep_pre = pd.Series(
        xgb_rep_pre_pw.feature_importances_, index=X_tr_pre_rep_sc.columns
    ).sort_values(ascending=False)

    print("\nTop 20 XGBoost (Augmented) — All Windows:")
    print(imp_xgb_aug_all.head(20).to_string())
    print("\nTop 20 XGBoost (Augmented) — Pre-onset Only:")
    print(imp_xgb_aug_pre.head(20).to_string())
    print("\nTop 20 XGBoost (Replace) — All Windows:")
    print(imp_xgb_rep_all.head(20).to_string())
    print("\nTop 20 XGBoost (Replace) — Pre-onset Only:")
    print(imp_xgb_rep_pre.head(20).to_string())

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n══════ Summary of All Results ══════")
    show_results('RF — Augmented',           [res_rf_aug_all, res_rf_aug_pre])
    show_results('RF — Replace',             [res_rf_rep_all, res_rf_rep_pre])
    show_results('XGBoost Augmented', [res_xgb_aug_all_pw, res_xgb_aug_pre_pw])
    show_results('XGBoost Replace',   [res_xgb_rep_all_pw, res_xgb_rep_pre_pw])

    # ═════════════════════════════════════════════════════════════════════════
    # SHAP
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n══════ SHAP Analysis ══════")

    mean_shap_aug_all, _ = run_shap(
        xgb_aug_all_pw, X_tr_all_aug_sc, X_te_all_aug_sc,
        'XGBoost Augmented All Windows', OUT
    )
    mean_shap_aug_pre, _ = run_shap(
        xgb_aug_pre_pw, X_tr_pre_aug_sc, X_te_pre_aug_sc,
        'XGBoost Augmented Pre-onset', OUT
    )
    mean_shap_rep_all, _ = run_shap(
        xgb_rep_all_pw, X_tr_all_rep_sc, X_te_all_rep_sc,
        'XGBoost Replace All Windows', OUT
    )
    mean_shap_rep_pre, _ = run_shap(
        xgb_rep_pre_pw, X_tr_pre_rep_sc, X_te_pre_rep_sc,
        'XGBoost Replace Pre-onset', OUT
    )

    run_shap_timestep(xgb_aug_all_pw, X_te_all_aug_sc, 'XGBoost Augmented All Windows', OUT)
    run_shap_timestep(xgb_aug_pre_pw, X_te_pre_aug_sc, 'XGBoost Augmented Pre-onset',   OUT)

    plot_trajectory_vs_raw_shap(mean_shap_aug_all, OUT, top_n=15)

    # ═════════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ═════════════════════════════════════════════════════════════════════════
    print("\nSaving visualisations...")

    plot_roc_pr(
        results_all=[
            (res_rf_aug_all,      'RF (Augmented)'),
            (res_xgb_aug_all_pw,  'XGBoost Augmented'),
            (res_xgb_rep_all_pw,  'XGBoost Replace'),
        ],
        results_pre=[
            (res_rf_aug_pre,      'RF (Augmented)'),
            (res_xgb_aug_pre_pw,  'XGBoost Augmented'),
            (res_xgb_rep_pre_pw,  'XGBoost Replace')
        ],
        y_test_all=y_te_all, y_test_pre=y_te_pre,
        suffix='trajectory', out_dir=OUT
    )

    plot_feature_importance({
        'RF Augmented — All Windows'  : imp_rf_aug_all,
        'RF Augmented — Pre-onset'    : imp_rf_aug_pre,
    }, top_n=20, suffix='rf_augmented', out_dir=OUT)

    plot_feature_importance({
        'RF Replace — All Windows'    : imp_rf_rep_all,
        'RF Replace — Pre-onset'      : imp_rf_rep_pre,
    }, top_n=20, suffix='rf_replace', out_dir=OUT)

    plot_feature_importance({
        'XGBoost Augmented — All Windows' : imp_xgb_aug_all,
        'XGBoost Augmented — Pre-onset'   : imp_xgb_aug_pre,
    }, top_n=20, suffix='xgb_augmented', out_dir=OUT)

    plot_feature_importance({
        'XGBoost Replace — All Windows'   : imp_xgb_rep_all,
        'XGBoost Replace — Pre-onset'     : imp_xgb_rep_pre,
    }, top_n=20, suffix='xgb_replace', out_dir=OUT)

    # baseline results for comparison plot — load from previous run
    try:
        with open(f'{base_dir}/rf_xgb_results/results_rf_xgb.pkl', 'rb') as f:
            prev = pickle.load(f)
        baseline_results = [
            prev['rf_all'], prev['rf_pre'],
            prev['res_xgb_all_scale'], prev['res_xgb_pre_scale'],
        ]
        for r in baseline_results:
            r['experiment'] = r['experiment'].replace('windows', 'Windows').replace('only', 'Only')
            if 'XGBoost' in r['model']:
                r['model'] = 'XGBoost'
            else:
                r['model'] = 'Random Forest'

        aug_results = [res_rf_aug_all, res_rf_aug_pre,
                       res_xgb_aug_all_pw, res_xgb_aug_pre_pw]
        rep_results = [res_rf_rep_all, res_rf_rep_pre,
                       res_xgb_rep_all_pw, res_xgb_rep_pre_pw]
        for r in aug_results + rep_results:
            r['experiment'] = r['experiment'].replace('(Augmented)', '').replace('(Replace)', '').strip()
            r['model'] = r['model'].split('(')[0].strip()

        plot_comparison_bar(baseline_results, aug_results, rep_results, OUT)
        print("  Comparison bar chart saved.")
    except FileNotFoundError:
        print("  Previous results not found — skipping comparison plot.")

    # ═════════════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════════════
    print("\nSaving models and results...")

    joblib.dump(rf_aug_all,      f'{RESULTS_OUT}/model_rf_aug_all.pkl')
    joblib.dump(rf_aug_pre,      f'{RESULTS_OUT}/model_rf_aug_pre.pkl')
    joblib.dump(rf_rep_all,      f'{RESULTS_OUT}/model_rf_rep_all.pkl')
    joblib.dump(rf_rep_pre,      f'{RESULTS_OUT}/model_rf_rep_pre.pkl')
    joblib.dump(xgb_aug_all_pw,  f'{RESULTS_OUT}/model_xgb_aug_all_pw.pkl')
    joblib.dump(xgb_aug_pre_pw,  f'{RESULTS_OUT}/model_xgb_aug_pre_pw.pkl')
    joblib.dump(xgb_rep_all_pw,  f'{RESULTS_OUT}/model_xgb_rep_all_pw.pkl')
    joblib.dump(xgb_rep_pre_pw,  f'{RESULTS_OUT}/model_xgb_rep_pre_pw.pkl')

    np.save(f'{RESULTS_OUT}/X_test_all_aug.npy', X_te_all_aug_sc.values)
    np.save(f'{RESULTS_OUT}/X_test_pre_aug.npy', X_te_pre_aug_sc.values)
    np.save(f'{RESULTS_OUT}/X_test_all_rep.npy', X_te_all_rep_sc.values)
    np.save(f'{RESULTS_OUT}/X_test_pre_rep.npy', X_te_pre_rep_sc.values)
    np.save(f'{RESULTS_OUT}/y_test_all.npy',     y_te_all)
    np.save(f'{RESULTS_OUT}/y_test_pre.npy',     y_te_pre)

    pd.Series(X_te_all_aug_sc.columns).to_csv(f'{RESULTS_OUT}/feature_names_aug.csv',  index=False)
    pd.Series(X_te_all_rep_sc.columns).to_csv(f'{RESULTS_OUT}/feature_names_rep.csv',  index=False)

    with open(f'{RESULTS_OUT}/results_trajectory.pkl', 'wb') as f:
        pickle.dump({
            'res_rf_aug_all'      : res_rf_aug_all,
            'res_rf_aug_pre'      : res_rf_aug_pre,
            'res_rf_rep_all'      : res_rf_rep_all,
            'res_rf_rep_pre'      : res_rf_rep_pre,
            'res_xgb_aug_all_pw'  : res_xgb_aug_all_pw,
            'res_xgb_aug_pre_pw'  : res_xgb_aug_pre_pw,
            'res_xgb_rep_all_pw'  : res_xgb_rep_all_pw,
            'res_xgb_rep_pre_pw'  : res_xgb_rep_pre_pw,
        }, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
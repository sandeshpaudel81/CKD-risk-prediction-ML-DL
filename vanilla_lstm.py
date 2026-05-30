# Import data processing and visualisation functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle

# Import model development and evaluation packages
from sklearn.metrics import (classification_report, confusion_matrix,
                                        roc_auc_score, average_precision_score,
                                        roc_curve, precision_recall_curve, f1_score, recall_score)
import torch
import torch.nn as nn

# Import utility functions
from utils.common import create_folder, show_results
from utils.lstm_bilstm import (
    build_windows_3d, scale_3d, smote_3d, make_loader
)
from utils.rf_xgb import find_best_threshold

# Set device for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

class VanillaLSTM(nn.Module):
    """
    Single-layer LSTM → last hidden state → Dropout → Linear → logit.
    Intentionally simple — serves as the LSTM baseline before
    adding bidirectionality, attention, or stacking.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
        # No sigmoid here — BCEWithLogitsLoss applies it internally
 
    def forward(self, x):
        out, _  = self.lstm(x)              # (batch, seq_len, hidden)
        last    = out[:, -1, :]             # last time step only
        last    = self.dropout(last)
        logits  = self.fc(last).squeeze(-1) # (batch,)
        return logits
    
def train_lstm(model, train_loader, val_loader, pos_weight,
               epochs=80, lr=1e-3, patience=12, device='cpu'):
 
    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
 
    best_auc       = 0.0
    best_state     = None
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': [], 'val_auc': []}
 
    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_loader.dataset)
 
        # --- validate ---
        model.eval()
        val_logits_all, val_labels_all = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                val_logits_all.append(logits.cpu())
                val_labels_all.append(yb.cpu())
        val_loss  /= len(val_loader.dataset)
        val_probs  = torch.sigmoid(torch.cat(val_logits_all)).numpy()
        val_labels = torch.cat(val_labels_all).numpy()
        val_auc    = roc_auc_score(val_labels, val_probs)
 
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        scheduler.step(val_auc)
 
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_AUC: {val_auc:.4f}")
 
        if val_auc > best_auc:
            best_auc       = val_auc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                break
 
    model.load_state_dict(best_state)
    return model, history

def evaluate_model(model, X_test, y_test, X_val, y_val,
                  model_name='Vanilla LSTM', experiment='All windows',
                  device='cpu'):
    model.eval()
    with torch.no_grad():
        val_probs  = torch.sigmoid(
            model(torch.tensor(X_val).to(device))).cpu().numpy()
        test_probs = torch.sigmoid(
            model(torch.tensor(X_test).to(device))).cpu().numpy()
 
    thresh = find_best_threshold(y_val, val_probs, method='youden')
 
    print(f"\n{'='*60}")
    print(f"  {model_name}  |  {experiment}")
    print(f"{'='*60}")
    print(f"  Optimal threshold (Youden's J on val): {thresh:.3f}")
 
    y_pred = (test_probs >= thresh).astype(int)
    print(f"\n{classification_report(y_test, y_pred, target_names=['No CKD','CKD'])}")
    print(f"  ROC-AUC : {roc_auc_score(y_test, test_probs):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y_test, test_probs):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
 
    return {
        'model'      : model_name,
        'experiment' : experiment,
        'threshold'  : thresh,
        'roc_auc'    : roc_auc_score(y_test, test_probs),
        'pr_auc'     : average_precision_score(y_test, test_probs),
        'f1_ckd'     : f1_score(y_test, y_pred),
        'recall_ckd' : recall_score(y_test, y_pred),
        'y_prob'     : test_probs,
        'y_pred'     : y_pred,
        'cm'         : cm,
    }

def train_and_evaluate(data, model_name, balance_method, experiment):
    print(f"\n Training Vanilla LSTM on {balance_method} data for {experiment} experiment..")
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    N_FEATURES = X_train.shape[2]
     
    pos_weight = torch.tensor(
        (y_train == 0).sum() / y_train.sum(),
        dtype=torch.float32
    )
    
    if (balance_method == 'pos_scaled'):
        print(f"  pos_weight: {pos_weight:.2f}  |  n_features: {N_FEATURES}")
     
    model = VanillaLSTM(input_size=N_FEATURES, hidden_size=64,
                                  num_layers=1, dropout=0.3)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
     
    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val, shuffle=False)
     
    model, history = train_lstm(
        model, train_loader, val_loader,
        pos_weight, epochs=80, lr=1e-3, patience=12, device=DEVICE)
     
    result = evaluate_model(
        model,
        X_test, y_test,
        X_val,  y_val,
        model_name=model_name, experiment=experiment, device=DEVICE)

    return model, history, result

def main(base_dir):
    create_folder(base_dir, ['plots/lstm', 'results/lstm'])

    PLOT_DIR = base_dir / 'plots/lstm'
    OUT_DIR  = base_dir / 'results/lstm'

    print("Vanilla LSTM modeling...")

    # Import train, validation and test dataframes from file
    train_df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/train_df.npy', allow_pickle=True),
        columns=pd.read_csv(f'{base_dir}/dataset/column_names.csv').iloc[:,0].tolist()
    )
    val_df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/val_df.npy', allow_pickle=True),
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        np.load(f'{base_dir}/dataset/test_df.npy', allow_pickle=True),
        columns=train_df.columns
    )

    WINDOW_SIZE  = 5
    TARGET_COL   = 'CKD'
    FEATURE_COLS = [c for c in train_df.columns if c not in ['patient_id']]
    print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # Build sliding windows for all-window experiment
    X_train_all, y_train_all, pid_train_all = build_windows_3d(train_df, FEATURE_COLS, TARGET_COL)
    X_val_all, y_val_all, pid_val_all = build_windows_3d(val_df, FEATURE_COLS, TARGET_COL)
    X_test_all, y_test_all, pid_test_all = build_windows_3d(test_df, FEATURE_COLS, TARGET_COL)

    # Build sliding windows for pre-onset-only experiment
    X_train_pre, y_train_pre, pid_train_pre = build_windows_3d(train_df, FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    X_val_pre, y_val_pre, pid_val_pre = build_windows_3d(val_df, FEATURE_COLS, TARGET_COL, pre_onset_only=True)
    X_test_pre, y_test_pre, pid_test_pre = build_windows_3d(test_df, FEATURE_COLS, TARGET_COL, pre_onset_only=True)

    print(f"\nAll-windows  — Train: {X_train_all.shape}  "
        f"CKD=1: {y_train_all.sum():.0f}")
    print(f"Pre-onset    — Train: {X_train_pre.shape}  "
        f"CKD=1: {y_train_pre.sum():.0f}")
    
    # Scale features for both experiments
    X_train_all, X_val_all, X_test_all, scaler_all = scale_3d(X_train_all, X_val_all, X_test_all)
    X_train_pre, X_val_pre, X_test_pre, scaler_pre = scale_3d(X_train_pre, X_val_pre, X_test_pre)
    
    # Apply SMOTE to training data for both experiments
    print("\nSMOTE for LSTM all-windows train...")
    X_train_all_sm, y_train_all_sm = smote_3d(X_train_all, y_train_all)
    print("SMOTE for LSTM pre-onset train...")
    X_train_pre_sm, y_train_pre_sm = smote_3d(X_train_pre, y_train_pre, k_neighbors=3)

    # Train and evaluate Vanilla LSTM on all-window data with pos_weight balancing
    data_all_scale = {
        'X_train': X_train_all,
        'y_train': y_train_all,
        'X_val': X_val_all,
        'y_val': y_val_all,
        'X_test': X_test_all,
        'y_test': y_test_all
    }
    model_all_scale, history_all_scale, res_all_scale = train_and_evaluate(
        data_all_scale, 'Vanilla LSTM (pos weight)', 'pos_scaled', 'All windows'
    )

    # Train and evaluate Vanilla LSTM on pre-onset-only data with pos_weight balancing
    data_pre_scale = {
        'X_train': X_train_pre,
        'y_train': y_train_pre,
        'X_val': X_val_pre,
        'y_val': y_val_pre,
        'X_test': X_test_pre,
        'y_test': y_test_pre
    }
    model_pre_scale, history_pre_scale, res_pre_scale = train_and_evaluate(
        data_pre_scale, 'Vanilla LSTM (pos weight)', 'pos_scaled', 'Pre-onset only'
    )

    # Train and evaluate Vanilla LSTM on all-window data with SMOTE balancing
    data_all_smote = {
        'X_train': X_train_all_sm,
        'y_train': y_train_all_sm,
        'X_val': X_val_all,
        'y_val': y_val_all,
        'X_test': X_test_all,
        'y_test': y_test_all
    }
    model_all_smote, history_all_smote, res_all_smote = train_and_evaluate(
        data_all_smote, 'Vanilla LSTM (SMOTE)', 'SMOTE', 'All windows'
    )

    # Train and evaluate Vanilla LSTM on pre-onset-only data with SMOTE balancing
    data_pre_smote = {
        'X_train': X_train_pre_sm,
        'y_train': y_train_pre_sm,
        'X_val': X_val_pre,
        'y_val': y_val_pre,
        'X_test': X_test_pre,
        'y_test': y_test_pre
    }
    model_pre_smote, history_pre_smote, res_pre_smote = train_and_evaluate(
        data_pre_smote, 'Vanilla LSTM (SMOTE)', 'SMOTE', 'Pre-onset only'
    )

    # Show summary of results
    show_results('Vanilla LSTM (pos_weight balanced)', [res_all_scale, res_pre_scale])
    show_results('Vanilla LSTM (SMOTE balanced)', [res_all_smote, res_pre_smote])

    # Result visualisation: Training curves, ROC and PR curves

    # Training curves for all experiments that shows model history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, hist, title in [
        (axes[0][0], history_all_scale, 'Vanilla LSTM — pos_weight - All windows'),
        (axes[0][1], history_pre_scale, 'Vanilla LSTM — pos_weight - Pre-onset only'),
        (axes[1][0], history_all_smote, 'Vanilla LSTM — SMOTE - Pre-onset only'),
        (axes[1][1], history_pre_smote, 'Vanilla LSTM — SMOTE - Pre-onset only')
    ]:
        ax2 = ax.twinx()
        ax.plot(hist['train_loss'], label='Train loss',  color='steelblue')
        ax.plot(hist['val_loss'],   label='Val loss',    color='cornflowerblue', linestyle='--')
        ax2.plot(hist['val_auc'],   label='Val AUC',     color='darkorange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss',     color='steelblue')
        ax2.set_ylabel('Val AUC', color='darkorange')
        ax.set_title(title)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/lstm_training_curves.png', dpi=150, bbox_inches='tight')

    # Precision-Recall Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, exp_label, y_test_exp, res_list in [
        (axes[0], 'All windows',    y_test_all, [res_all_scale, res_all_smote]),
        (axes[1], 'Pre-onset only', y_test_pre, [res_pre_scale, res_pre_smote]),
    ]:
        for r in res_list:
            prec, rec, _ = precision_recall_curve(y_test_exp, r['y_prob'])
            ax.plot(rec, prec, lw=2,
                    label=f"{r['model']} (PR-AUC={r['pr_auc']:.3f})")
        baseline = y_test_exp.mean()
        ax.axhline(baseline, color='k', linestyle='--', lw=1,
                label=f'Baseline ({baseline:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — {exp_label}')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/lstm_pr_curves.png', dpi=150, bbox_inches='tight')

    # ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, exp_label, y_test_exp, res_list in [
        (axes[0], 'All windows',    y_test_all, [res_all_scale, res_all_smote]),
        (axes[1], 'Pre-onset only', y_test_pre, [res_pre_scale, res_pre_smote]),
    ]:
        for r in res_list:
            fpr, tpr, _ = roc_curve(y_test_exp, r['y_prob'])
            ax.plot(fpr, tpr, lw=2,
                    label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {exp_label}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/roc_all_models.png', dpi=150, bbox_inches='tight')

    # Save models
    joblib.dump(model_all_scale, f'{OUT_DIR}/model_lstm_all_scale.pkl')
    joblib.dump(model_pre_scale, f'{OUT_DIR}/model_lstm_pre_scale.pkl')
    joblib.dump(model_all_smote, f'{OUT_DIR}/model_lstm_all_smote.pkl')
    joblib.dump(model_pre_smote, f'{OUT_DIR}/model_lstm_pre_smote.pkl')

    # Save test data + predictions
    np.save(f'{OUT_DIR}/X_test_all.npy',    X_test_all)
    np.save(f'{OUT_DIR}/y_test_all.npy',    y_test_all)
    np.save(f'{OUT_DIR}/X_train_all.npy',   X_train_all)
    np.save(f'{OUT_DIR}/y_train_all.npy',   y_train_all)
    np.save(f'{OUT_DIR}/X_test_pre.npy',    X_test_pre)
    np.save(f'{OUT_DIR}/y_test_pre.npy',    y_test_pre)
    np.save(f'{OUT_DIR}/X_train_pre.npy',   X_train_pre)
    np.save(f'{OUT_DIR}/y_train_pre.npy',   y_train_pre)

    # Save results dict
    with open(f'{OUT_DIR}/results_lstm.pkl', 'wb') as f:
        pickle.dump({'all_scale': res_all_scale, 'pre_scale': res_pre_scale,
                    'all_smote': res_all_smote, 'pre_smote': res_pre_smote}, f)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                                        roc_auc_score, average_precision_score,
                                        roc_curve, precision_recall_curve, f1_score, recall_score)
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# def build_windows_3d(data_df, feature_cols, target_col,
#                      window_size=5, pre_onset_only=False):
#     X_list, y_list = [], []
#     for pid, group in data_df.groupby('patient_id'):
#         group    = group.sort_values('diabetic_year')
#         features = group[feature_cols].values
#         labels   = group[target_col].values
#         n        = len(features)
#         for i in range(n - window_size):
#             window_feats  = features[i : i + window_size]
#             window_labels = labels[i : i + window_size]
#             target        = labels[i + window_size]
#             if pre_onset_only and window_labels.sum() > 0:
#                 continue
#             X_list.append(window_feats)
#             y_list.append(target)
#     return (np.array(X_list, dtype=np.float32),
#             np.array(y_list, dtype=np.float32))

def build_windows_3d(data_df, feature_cols, target_col,
                     window_size=5, pre_onset_only=False):
    X_list, y_list, pid_list = [], [], []
    for pid, group in data_df.groupby('patient_id'):
        group    = group.sort_values('diabetic_year')
        features = group[feature_cols].values
        labels   = group[target_col].values
        n        = len(features)
 
        for i in range(n - window_size):
            window_feats  = features[i : i + window_size]
            window_labels = labels[i : i + window_size]
            target        = labels[i + window_size]
 
            if pre_onset_only and window_labels.sum() > 0:
                continue
 
            X_list.append(window_feats)
            y_list.append(target)
            pid_list.append(pid)
 
    return (np.array(X_list,   dtype=np.float32),
            np.array(y_list,   dtype=np.float32),
            np.array(pid_list, dtype=np.int32))

def scale_3d(X_tr, X_v, X_te):
    n_feat  = X_tr.shape[2]
    sc      = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_v_sc  = sc.transform(X_v.reshape(-1, n_feat)).reshape(X_v.shape)
    X_te_sc = sc.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)
    return X_tr_sc, X_v_sc, X_te_sc, sc

def smote_3d(X_train, y_train, k_neighbors=5):
    shape  = X_train.shape
    X_flat = X_train.reshape(len(X_train), -1)
    k      = min(k_neighbors, int(y_train.sum()) - 1)
    sm     = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_flat, y_train)
    X_3d   = X_res.reshape(-1, shape[1], shape[2])
    print(f"  After SMOTE — CKD=0: {(y_res==0).sum()}  CKD=1: {(y_res==1).sum()}")
    return X_3d.astype(np.float32), y_res.astype(np.float32)

def make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
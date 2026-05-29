import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, f1_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def scale_splits(X_train, X_val, X_test):
    scaler   = StandardScaler()
    X_train_sc  = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_sc, X_val_sc, X_test_sc, scaler

def apply_smote(X_train, y_train, k_neighbors=5):
    k = min(k_neighbors, int(y_train.sum()) - 1)
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  Before SMOTE — CKD=0: {(y_train==0).sum()}  CKD=1: {(y_train==1).sum()}")
    print(f"   After SMOTE — CKD=0: {(y_res==0).sum()}  CKD=1: {(y_res==1).sum()}")
    return X_res, y_res

def find_best_threshold(y_true, y_prob, method='youden'):
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_thresh = thresholds[np.argmax(j_scores)]
    else:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresh = thresholds[np.argmax(f1[:-1])]
    return float(best_thresh)

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

def rf_fit_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, experiment):
    param_grid = {
        'n_estimators'      : [200, 300, 500],
        'max_depth'         : [None, 10, 20, 30],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf'  : [1, 2, 4],
        'max_features'      : ['sqrt', 'log2'],
        'class_weight'      : ['balanced', {0:1, 1:3}, {0:1, 1:5}]
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=30, scoring='roc_auc', cv=3,
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    print(f"  Best RF params : {search.best_params_}")
    print(f"  Best CV AUC    : {search.best_score_:.4f}")
    best = search.best_estimator_
    res  = evaluate(best, X_test, y_test, X_val, y_val,
                    model_name='Random Forest', experiment=experiment)
    return best, res
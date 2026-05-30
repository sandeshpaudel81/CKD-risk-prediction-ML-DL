import os
import pandas as pd

def create_folder(base_dir, folders):
    for folder in folders:
        os.makedirs(f'{base_dir}/{folder}', exist_ok=True)

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
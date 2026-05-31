import sys
from pathlib import Path

import preprocessing
import shap_analysis
import statistical_analysis
import rf_xgboost
import vanilla_lstm
import bilstm_attention
import trajectory_analysis
import causal_analysis
import converter_clustering

def run_preprocessing(base_dir):
    print("Starting data preprocessing...")
    preprocessing.main(base_dir)

def run_statistical(base_dir):
    print("Statistical analysis of the dataset...")
    statistical_analysis.main(base_dir)

def run_rf_xgboost(base_dir):
    print("Random Forest and XGBoost modeling...")
    rf_xgboost.main(base_dir)

def run_lstm(base_dir):
    print("Vanilla LSTM modeling...")
    vanilla_lstm.main(base_dir)

def run_bilstm_attention(base_dir):
    print("BiLSTM Attention modeling...")
    bilstm_attention.main(base_dir)

def run_shap(base_dir):
    print("SHAP analysis of the trained XGBoost models...")
    shap_analysis.main(base_dir)

def run_trajectory(base_dir):
    print("Trajectory analysis of CKD progression...")
    trajectory_analysis.main(base_dir)

def run_causal(base_dir):
    print("Causal inference analysis of CKD progression...")
    causal_analysis.main(base_dir)

def run_clustering(base_dir):
    print("Converter clustering analysis...")
    converter_clustering.main(base_dir)


# Checks for existing results to avoid redundant computations

def has_clean_data(base_dir):
    return (base_dir / "dataset" / "clean_data.npy").exists()

def has_train_val_test(base_dir):
    dataset_dir = base_dir / "dataset"
    required = [
        dataset_dir / "train_df.npy",
        dataset_dir / "val_df.npy",
        dataset_dir / "test_df.npy",
    ]
    return all(f.exists() for f in required)

def shap_results_exist(base_dir):
    shap_dir = base_dir / "results" / "shap"
    return shap_dir.exists() and any(shap_dir.iterdir())

def rf_xgb_results_exist(base_dir):
    rf_dir = base_dir / "results" / "rf_xgb"
    return rf_dir.exists() and any(rf_dir.iterdir())

# Dependency checks

def ensure_clean_data(base_dir):
    if not has_clean_data(base_dir):
        run_preprocessing(base_dir)

def ensure_train_val_test(base_dir):
    if not has_train_val_test(base_dir):
        run_preprocessing(base_dir)

def ensure_rf_xgb(base_dir):
    if not rf_xgb_results_exist(base_dir):
        ensure_train_val_test(base_dir)
        run_rf_xgboost(base_dir)

def ensure_shap(base_dir):
    if not shap_results_exist(base_dir):
        ensure_rf_xgb(base_dir)
        run_shap(base_dir)

# Main execution flow

def run_all(base_dir):
    run_preprocessing(base_dir)
    run_statistical(base_dir)
    run_rf_xgboost(base_dir)
    run_lstm(base_dir)
    run_bilstm_attention(base_dir)
    run_shap(base_dir)
    run_trajectory(base_dir)
    run_causal(base_dir)
    run_clustering(base_dir)

def main():
    base_dir = Path(__file__).resolve().parent

    # Run everything if no specific task is given
    if len(sys.argv) == 1:
        run_all(base_dir)
        return

    task = sys.argv[1].lower()

    if task == "preprocessing":
        run_preprocessing(base_dir)

    elif task == "statistical":
        ensure_clean_data(base_dir)
        run_statistical(base_dir)

    elif task == "rf_xgboost":
        ensure_train_val_test(base_dir)
        run_rf_xgboost(base_dir)

    elif task == "lstm":
        ensure_train_val_test(base_dir)
        run_lstm(base_dir)

    elif task == "bilstm_attention":
        ensure_train_val_test(base_dir)
        run_bilstm_attention(base_dir)

    elif task == "shap":
        ensure_shap(base_dir)

    elif task == "trajectory":
        ensure_shap(base_dir)
        run_trajectory(base_dir)

    elif task == "causal":
        ensure_clean_data(base_dir)
        ensure_shap(base_dir)
        run_causal(base_dir)

    elif task == "clustering":
        ensure_clean_data(base_dir)
        run_clustering(base_dir)

    else:
        print(
            "Unknown task. Choose from:\n"
            "preprocessing\n"
            "statistical\n"
            "rf_xgboost\n"
            "lstm\n"
            "bilstm_attention\n"
            "shap\n"
            "trajectory\n"
            "causal\n"
            "clustering"
        )

if __name__ == "__main__":
    main()
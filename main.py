import preprocessing
import shap_analysis
import statistical_analysis
import rf_xgboost
import vanilla_lstm
import bilstm_attention
import trajectory_analysis
import causal_analysis
import converter_clustering
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent

    # print("Starting data preprocessing...")
    # preprocessing.main(base_dir)

    # print("Statistical analysis of the dataset...")
    # statistical_analysis.main(base_dir)

    # print("Random Forest and XGBoost modeling...")
    # rf_xgboost.main(base_dir)

    # print("Vanilla LSTM modeling...")
    # vanilla_lstm.main(base_dir)

    # print("Bilstm Attention modeling...")
    # bilstm_attention.main(base_dir)

    # print("SHAP analysis of the trained XGBoost models...")
    # shap_analysis.main(base_dir)

    # print("Trajectory analysis of CKD progression...")
    # trajectory_analysis.main(base_dir)

    # print("Causal inference analysis of CKD progression...")
    # causal_analysis.main(base_dir)

    print("Converter clustering analysis...")
    converter_clustering.main(base_dir)

if __name__ == "__main__":
    main()

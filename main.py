import preprocessing
import shap_analysis
import statistical_analysis
import rf_xgboost
import vanilla_lstm
import bilstm_attention

def main():
    print("Starting data preprocessing...")
    preprocessing.main()

    print("Statistical analysis of the dataset...")
    statistical_analysis.main()

    print("Random Forest and XGBoost modeling...")
    rf_xgboost.main()

    print("Vanilla LSTM modeling...")
    vanilla_lstm.main()

    print("Bilstm Attention modeling...")
    bilstm_attention.main()

    print("SHAP analysis of the trained XGBoost models...")
    shap_analysis.main()

if __name__ == "__main__":
    main()

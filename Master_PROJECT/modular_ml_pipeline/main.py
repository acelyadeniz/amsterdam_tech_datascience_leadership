
import os
from data_handling.loader import load_and_preprocess_data
from analysis.eda import perform_eda
from models.logistic_regression import train_logistic_regression
from models.xgboost_model import train_xgboost
from models.dnn_pytorch import train_dnn
from evaluation.plotting import plot_model_comparison_roc

def main():
    """
    Runs the complete machine learning pipeline for the SUSY analysis.
    """
    # --- 1. Configuration ---
    CONFIG = {
        "n_rows": 1000,       # Reduce row count for a quick run
        "test_size": 0.2,
        "random_state": 42,
        "run_eda": True,         # Set to True to run the EDA
        "results_dir": "results"
    }
    
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # --- 2. Data Loading and Preprocessing ---
    X_train, X_test, y_train, y_test, df_full, feature_sets = load_and_preprocess_data(
        n_rows=CONFIG["n_rows"],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

  
    if CONFIG["run_eda"]:
        perform_eda(df_full, output_dir=os.path.join(CONFIG["results_dir"], "eda_plots"))

  
    model_results = {}

 
    # Model 1: Logistic Regression (All Features)
    X_train_all, X_test_all = feature_sets["all"]
    lr_model, lr_auc, lr_preds = train_logistic_regression(X_train_all, y_train, X_test_all, y_test)
    model_results['Logistic Regression'] = (y_test, lr_preds, lr_auc)
    
    # Model 2: XGBoost (All Features)
    xgb_model, xgb_auc, xgb_preds = train_xgboost(X_train_all, y_train, X_test_all, y_test)
    model_results['XGBoost'] = (y_test, xgb_preds, xgb_auc)

    # Model 3: PyTorch DNN (All Features)

    dnn_model, dnn_auc, dnn_preds = train_dnn(X_train_all, y_train, X_test_all, y_test, epochs=5)
    model_results['PyTorch DNN'] = (y_test, dnn_preds, dnn_auc)

    # Optional: Test models with different feature sets
    print("\n--- Bonus Analysis: XGBoost (High-Level Features Only) ---")
    X_train_high, X_test_high = feature_sets["high_level"]
    xgb_high_model, xgb_high_auc, xgb_high_preds = train_xgboost(X_train_high, y_train, X_test_high, y_test)
    model_results['XGBoost (High-Level Feats)'] = (y_test, xgb_high_preds, xgb_high_auc)

    # --- 5. Model Comparison ---
    print("\n--- Performance Summary of All Models ---")
    for name, (_, _, auc_score) in model_results.items():
        print(f"- {name}: Test AUC = {auc_score:.4f}")
        
    plot_model_comparison_roc(model_results, output_dir=CONFIG["results_dir"])
    
    print("\n>>> Analysis Pipeline Completed Successfully! <<<")


if __name__ == "__main__":
    main()
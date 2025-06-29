# models/xgboost_model.py

import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
import time

def train_xgboost(X_train, y_train, X_test, y_test, ):
    """
    Trains and evaluates an XGBoost model.
    """
    print("--- Starting XGBoost Model Training ---")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    fit_params = {}

    start_time = time.time()
    model.fit(X_train, y_train, **fit_params)
    end_time = time.time()
    
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Test Set ROC AUC: {test_auc:.4f}")
    
    print("--- XGBoost Model Training Complete ---\n")
    return model, test_auc, y_pred_proba
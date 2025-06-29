# models/logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Trains a Logistic Regression model with GridSearchCV and returns the best model.
    """
    print("--- Starting Logistic Regression Model Training ---")
    
    param_grid = [
        {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear'], 'max_iter': [2000]},
        {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [2000]}
    ]

    lr = LogisticRegression(random_state=42)
    
    grid_search = GridSearchCV(
        lr,
        param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC AUC score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test set ROC AUC score: {test_auc:.4f}")
    
    print("--- Logistic Regression Model Training Complete ---\n")
    return best_model, test_auc, y_pred_proba
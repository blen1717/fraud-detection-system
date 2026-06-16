"""
Hyperparameter Tuning for XGBoost
Author: Blen Assefa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
import xgboost as xgb
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(dataset_name):
    """Load preprocessed data."""
    try:
        X_train = pd.read_csv(f'data/processed/{dataset_name}_X_train_scaled.csv')
        y_train = pd.read_csv(f'data/processed/{dataset_name}_y_train.csv').iloc[:, 0]
        X_test = pd.read_csv(f'data/processed/{dataset_name}_X_test_scaled.csv')
        y_test = pd.read_csv(f'data/processed/{dataset_name}_y_test.csv').iloc[:, 0]
        logger.info(f"Loaded {dataset_name} data: train {X_train.shape}, test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def tune_xgboost(X_train, y_train, dataset_name):
    """Perform hyperparameter tuning for XGBoost."""
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=cv, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info(f"Starting GridSearchCV for {dataset_name}")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters for {dataset_name}: {grid_search.best_params_}")
    logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # Save tuned model
    joblib.dump(grid_search.best_estimator_, f'models/xgb_tuned_{dataset_name}.pkl')
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_tuned_model(model, X_test, y_test, dataset_name):
    """Evaluate tuned model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_proba)
    
    logger.info(f"Tuned XGBoost {dataset_name} - F1: {f1:.4f}, AUC-PR: {auc_pr:.4f}")
    
    return f1, auc_pr

def main():
    logger.info("Starting hyperparameter tuning")
    
    results = []
    
    for dataset in ['ecom', 'credit']:
        try:
            X_train, X_test, y_train, y_test = load_data(dataset)
            model, best_params = tune_xgboost(X_train, y_train, dataset)
            f1, auc_pr = evaluate_tuned_model(model, X_test, y_test, dataset)
            
            results.append({
                'Dataset': dataset,
                'Best F1': f1,
                'Best AUC-PR': auc_pr,
                'Parameters': best_params
            })
        except Exception as e:
            logger.error(f"Tuning failed for {dataset}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    for r in results:
        print(f"\nDataset: {r['Dataset']}")
        print(f"Best F1: {r['Best F1']:.4f}")
        print(f"Best AUC-PR: {r['Best AUC-PR']:.4f}")
        print(f"Parameters: {r['Parameters']}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load data (squeeze not needed; just take the first column)
    X_train = pd.read_csv('data/processed/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]   # first column as Series
    X_test = pd.read_csv('data/processed/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]
    
    print("=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:,1]
    print(f"F1: {f1_score(y_test, y_pred_lr):.4f}")
    print(f"AUC-PR: {average_precision_score(y_test, y_proba_lr):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    
    print("\n=== XGBoost ===")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:,1]
    print(f"F1: {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"AUC-PR: {average_precision_score(y_test, y_proba_xgb):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
    
    # Cross-validation on XGBoost
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='f1')
    cv_auc_pr = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='average_precision')
    print(f"\nXGBoost 5-fold CV F1: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"XGBoost 5-fold CV AUC-PR: {cv_auc_pr.mean():.4f} (+/- {cv_auc_pr.std():.4f})")
    
    # Save best model
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    print("Model saved to models/xgb_model.pkl")

if __name__ == "__main__":
    main()

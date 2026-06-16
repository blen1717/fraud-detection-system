"""
Credit Card Data Preprocessing Pipeline
Author: Blen Assefa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_credit_data(file_path='data/raw/creditcard.csv'):
    """Load and clean credit card data."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded credit card data: {df.shape}")
        df.drop_duplicates(inplace=True)
        logger.info(f"After removing duplicates: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading credit data: {e}")
        raise

def scale_credit_features(X_train, X_test):
    """Scale Amount and Time features."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale Amount and Time
    X_train_scaled[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test_scaled[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])
    
    logger.info("Credit card features scaled (fit on train only)")
    return X_train_scaled, X_test_scaled, scaler

def main():
    logger.info("Starting credit card preprocessing pipeline")
    
    # Load data
    df = load_credit_data()
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_credit_features(X_train, X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"SMOTE: before {y_train.value_counts().to_dict()} -> after {y_train_res.value_counts().to_dict()}")
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train_res.to_csv('data/processed/credit_X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('data/processed/credit_X_test_scaled.csv', index=False)
    pd.Series(y_train_res).to_csv('data/processed/credit_y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/credit_y_test.csv', index=False)
    joblib.dump(scaler, 'models/credit_scaler.pkl')
    
    logger.info("Credit card preprocessing completed successfully")

if __name__ == "__main__":
    main()

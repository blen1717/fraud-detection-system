"""
Fraud Detection Preprocessing Pipeline with Scaling & Error Handling
Author: Blen Assefa
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {file_path} - shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def add_geolocation(fraud_df, ip_df):
    try:
        fraud_df = fraud_df.copy()
        ip_df = ip_df.copy()
        fraud_df['ip_int'] = fraud_df['ip_address'].astype('int64')
        ip_df['lower_int'] = ip_df['lower_bound_ip_address'].astype('int64')
        ip_df['upper_int'] = ip_df['upper_bound_ip_address'].astype('int64')
        ip_df_sorted = ip_df.sort_values('lower_int')
        fraud_df_sorted = fraud_df.sort_values('ip_int')
        merged = pd.merge_asof(fraud_df_sorted, ip_df_sorted[['lower_int', 'upper_int', 'country']],
                               left_on='ip_int', right_on='lower_int', direction='forward')
        merged = merged[merged['ip_int'] <= merged['upper_int']]
        merged = merged.drop(columns=['lower_int', 'upper_int']).sort_index()
        logger.info(f"Geolocation merged: {merged.shape[0]} rows")
        return merged
    except Exception as e:
        logger.error(f"Geolocation failed: {e}")
        raise

def engineer_features(df):
    df = df.copy()
    df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek
    df['device_count'] = df.groupby('device_id')['user_id'].transform('nunique')
    logger.info("Feature engineering done")
    return df

def scale_features(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    logger.info("Numerical features scaled (fit on train only)")
    return X_train_scaled, X_test_scaled, scaler

def main():
    logger.info("Starting preprocessing pipeline")
    fraud = load_data('data/raw/Fraud_Data.csv')
    ip = load_data('data/raw/IpAddress_to_Country.csv')
    credit = load_data('data/raw/creditcard.csv')  # not used further here but kept
    
    fraud.drop_duplicates(inplace=True)
    fraud['signup_time'] = pd.to_datetime(fraud['signup_time'])
    fraud['purchase_time'] = pd.to_datetime(fraud['purchase_time'])
    credit.drop_duplicates(inplace=True)
    
    fraud = add_geolocation(fraud, ip)
    fraud = engineer_features(fraud)
    
    feature_cols = ['purchase_value', 'age', 'time_since_signup_hours', 'purchase_hour',
                    'purchase_dayofweek', 'device_count', 'source', 'browser', 'sex', 'country']
    X = fraud[feature_cols].copy()
    y = fraud['class']
    
    X = pd.get_dummies(X, columns=['source', 'browser', 'sex', 'country'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    numeric_cols = ['purchase_value', 'age', 'time_since_signup_hours', 'purchase_hour', 'purchase_dayofweek', 'device_count']
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numeric_cols)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"SMOTE: before {y_train.value_counts().to_dict()} -> after {y_train_res.value_counts().to_dict()}")
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    X_train_res.to_csv('data/processed/X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index=False)
    pd.Series(y_train_res).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("Preprocessing done. Saved scaled data and scaler.")

if __name__ == "__main__":
    main()

import pytest
import pandas as pd
import os

def test_data_exists():
    """Check that raw data files exist."""
    assert os.path.exists('data/raw/Fraud_Data.csv')
    assert os.path.exists('data/raw/creditcard.csv')

def test_preprocessed_data_exists():
    """Check that preprocessed files exist."""
    assert os.path.exists('data/processed/X_train_scaled.csv')
    assert os.path.exists('data/processed/y_train.csv')

def test_columns_present():
    """Check that expected columns exist."""
    df = pd.read_csv('data/processed/X_train_scaled.csv')
    expected_cols = ['purchase_value', 'age', 'time_since_signup_hours', 
                     'purchase_hour', 'purchase_dayofweek', 'device_count']
    for col in expected_cols:
        assert col in df.columns

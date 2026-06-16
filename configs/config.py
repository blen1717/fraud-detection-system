"""
Centralized configuration for fraud detection pipeline
Author: Blen Assefa
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(file)))

# Data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed')

# Model and report paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# ============================================================
# MODEL PARAMETERS
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================
# XGBOOST BASE PARAMETERS
# ============================================================
XGB_PARAMS = {
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# ============================================================
# HYPERPARAMETER TUNING GRID
# ============================================================
TUNING_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# ============================================================
# FEATURE COLUMNS
# ============================================================
ECOMMERCE_NUMERIC_COLS = [
    'purchase_value', 'age', 'time_since_signup_hours',
    'purchase_hour', 'purchase_dayofweek', 'device_count'
]

ECOMMERCE_CATEGORICAL_COLS = ['source', 'browser', 'sex', 'country']
CREDIT_NUMERIC_COLS = ['Amount', 'Time']

# ============================================================
# LOGGING
# ============================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

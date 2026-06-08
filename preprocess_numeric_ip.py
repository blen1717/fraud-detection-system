
"""
Complete preprocessing for Fraud Detection - Numeric IP version
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("Starting Preprocessing Pipeline (Numeric IP)")
print("="*50)

# Load data
fraud_df = pd.read_csv('/content/data/raw/Fraud_Data.csv')
ip_df = pd.read_csv('/content/data/raw/IpAddress_to_Country.csv')
credit_df = pd.read_csv('/content/data/raw/creditcard.csv')

print(f"\nFraud Data shape: {fraud_df.shape}")
print(f"IP Data shape: {ip_df.shape}")
print(f"Credit Data shape: {credit_df.shape}")

# Data Cleaning
fraud_df.drop_duplicates(inplace=True)
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

credit_df.drop_duplicates(inplace=True)

print(f"\nAfter cleaning - Fraud shape: {fraud_df.shape}, Credit shape: {credit_df.shape}")

# Class Imbalance Analysis
print("\n" + "="*50)
print("CLASS IMBALANCE ANALYSIS")
print("="*50)
print("E-Commerce:")
print(fraud_df['class'].value_counts(normalize=True))
print("\nCredit Card:")
print(credit_df['Class'].value_counts(normalize=True))

# Geolocation with numeric IPs
# Convert IP columns to integers (they are floats in CSV)
fraud_df['ip_int'] = fraud_df['ip_address'].astype('int64')
ip_df['lower_int'] = ip_df['lower_bound_ip_address'].astype('int64')
ip_df['upper_int'] = ip_df['upper_bound_ip_address'].astype('int64')

# Sort for merge_asof
ip_df_sorted = ip_df.sort_values('lower_int')
fraud_df_sorted = fraud_df.sort_values('ip_int')

# Merge using asof (find the range where ip_int falls between lower and upper)
temp = pd.merge_asof(
    fraud_df_sorted,
    ip_df_sorted[['lower_int', 'upper_int', 'country']],
    left_on='ip_int',
    right_on='lower_int',
    direction='forward'
)

# Keep only rows where ip_int <= upper_int
temp = temp[temp['ip_int'] <= temp['upper_int']]
fraud_df = temp.drop(columns=['lower_int', 'upper_int']).sort_index()

print("\n" + "="*50)
print("TOP FRAUD COUNTRIES")
print("="*50)
if 'country' in fraud_df.columns and fraud_df['class'].sum() > 0:
    print(fraud_df[fraud_df['class']==1]['country'].value_counts().head(10))
else:
    print("No fraud cases or country column missing after merge.")

# Feature Engineering
fraud_df['time_since_signup_hours'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
fraud_df['purchase_hour'] = fraud_df['purchase_time'].dt.hour
fraud_df['purchase_dayofweek'] = fraud_df['purchase_time'].dt.dayofweek
fraud_df['device_count'] = fraud_df.groupby('device_id')['user_id'].transform('nunique')

print("\n" + "="*50)
print("NEW FEATURES ADDED")
print("="*50)
print(['time_since_signup_hours', 'purchase_hour', 'purchase_dayofweek', 'device_count'])

# Prepare for modeling (fraud data only)
# Select features and target
feature_cols = ['purchase_value', 'age', 'time_since_signup_hours', 'purchase_hour', 
                'purchase_dayofweek', 'device_count', 'source', 'browser', 'sex']
if 'country' in fraud_df.columns:
    feature_cols.append('country')

X = fraud_df[feature_cols].copy()
y = fraud_df['class']

# Fill any missing values (should be none, but safe)
X = X.fillna('Unknown')

# One-hot encode categorical columns
categorical_cols = ['source', 'browser', 'sex']
if 'country' in X.columns:
    categorical_cols.append('country')
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Feature matrix shape after encoding: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n" + "="*50)
print("SMOTE RESULTS")
print("="*50)
print(f"Before SMOTE - Train class distribution:\n{y_train.value_counts()}")
print(f"\nAfter SMOTE - Train class distribution:\n{y_train_res.value_counts()}")
print("\n" + "="*50)
print("Preprocessing completed successfully!")
print("="*50)

# Save processed data
import os
os.makedirs('/content/data/processed', exist_ok=True)
fraud_df.to_csv('/content/data/processed/fraud_processed.csv', index=False)
credit_df.to_csv('/content/data/processed/credit_processed.csv', index=False)
print("\nSaved processed datasets to /content/data/processed/")
print("Shape of final feature matrix:", X.shape)

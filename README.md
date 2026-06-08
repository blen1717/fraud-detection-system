# Fraud Detection System

Advanced Cross-Platform Machine Learning for E‑commerce & Banking Transactions

This project builds a unified fraud detection engine for Adey Innovations Inc., capable of monitoring two separate transaction pipelines:
- E‑commerce – rich user, device, behavioural context.
- Bank credit card – anonymised PCA‑transformed features.

The system detects fraudulent transactions in real time, minimises false positives (customer friction) and false negatives (financial loss), and provides explainable predictions using SHAP.

---

## 📌 Core Objectives

- Clean, preprocess, and merge multi‑source datasets (IP geolocation, temporal logs).
- Engineer behavioural features: transaction velocity, time since signup, hour‑of‑day, device multiplicity.
- Handle extreme class imbalance using SMOTE only on training folds.
- Train and compare Logistic Regression (baseline) and XGBoost (ensemble) using imbalanced‑aware metrics: AUC‑PR, F1‑score, confusion matrix.
- Explain model decisions with SHAP (global feature importance + individual force plots).
- Translate findings into actionable business recommendations.

---

## 📊 Datasets

| Dataset | Description | Records | Key Fields |
|---------|-------------|---------|-------------|
| Fraud_Data.csv | E‑commerce transactions | 151,112 | user_id, signup/purchase time, value, device_id, browser, IP, class (0/1) |
| IpAddress_to_Country.csv | IP range → country mapping | ~200,000 | lower/upper IP bounds (numeric), country |
| creditcard.csv | Bank credit card transactions | 284,807 | Time, V1–V28 (PCA), Amount, Class (0/1) |

Both target datasets are highly imbalanced:
- E‑commerce: ~9.4% fraud
- Credit card: ~0.17% fraud

---

## 🏗️ Project Structure

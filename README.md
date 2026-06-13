# Fraud Detection System

End-to-end machine learning solution for e‑commerce and credit card fraud detection

This project provides a production‑ready fraud detection pipeline for Adey Innovations Inc., handling two transaction streams:
- E‑commerce – rich user, device, and behavioural context.
- Credit card – anonymised PCA‑transacted features.

The system includes:
- Data cleaning, geolocation enrichment (IP to country), and feature engineering.
- Handling of extreme class imbalance using SMOTE (training set only).
- Logistic Regression baseline and XGBoost ensemble.
- Evaluation with imbalanced‑aware metrics: AUC‑PR, F1‑score, confusion matrix.
- SHAP explainability: global feature importance + individual force plots.
- Actionable business recommendations derived from SHAP insights.



## 🚀 Quick Start

### 1. Clone the repository
`bash
git clone https://github.com/blen1717/fraud-detection-system.git
cd fraud-detection-system

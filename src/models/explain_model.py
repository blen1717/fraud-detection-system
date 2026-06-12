import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs('reports', exist_ok=True)

# Load data
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
model = joblib.load('models/xgb_model.pkl')

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('reports/shap_summary.png', bbox_inches='tight')
plt.close()

# Load true labels
y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]
y_pred = model.predict(X_test)

# Find indices of each case
tp_mask = (y_test == 1) & (y_pred == 1)
fp_mask = (y_test == 0) & (y_pred == 1)
fn_mask = (y_test == 1) & (y_pred == 0)

# Helper to get first row index where mask is True
def get_first_row(mask):
    indices = mask[mask].index
    if len(indices) > 0:
        return indices[0]
    return None

# Generate force plots for each case
for name, mask in [('true_positive', tp_mask), ('false_positive', fp_mask), ('false_negative', fn_mask)]:
    idx = get_first_row(mask)
    if idx is not None:
        # Extract the SHAP values for that single row
        shap_single = shap_values[idx]
        shap_exp = explainer.expected_value
        # Force plot
        shap.force_plot(shap_exp, shap_single, X_test.loc[idx], matplotlib=True, show=False)
        plt.savefig(f'reports/shap_force_{name}.png', bbox_inches='tight')
        plt.close()
        print(f"Saved {name} force plot (row index {idx}).")
    else:
        print(f"No {name} example found.")

print("\nAll SHAP plots saved in 'reports/' directory.")

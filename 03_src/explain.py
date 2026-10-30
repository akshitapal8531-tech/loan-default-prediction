import os
import joblib
import matplotlib
matplotlib.use("Agg")  # draw to a file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from data_processing import load_data, preprocess
from train import engineer_features

# Build paths from this file's location, so it works from any folder
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rebuild the test data the same way training did
df = preprocess(load_data(os.path.join(BASE, "01_data", "Loan_default.csv")))
df = engineer_features(df)
X = df.drop(columns=["Default"])
y = df["Default"]
_, X_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the saved model and keep only the columns it was trained on
model = joblib.load(os.path.join(BASE, "04_models", "loan_default_model.joblib"))
cols = list(model.feature_names_in_)
X_sample = X_test.sample(500, random_state=42)[cols].astype(float)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Text summary: which features matter most on average
importance = np.abs(shap_values).mean(axis=0)
order = np.argsort(importance)[::-1]
print("Top features by average impact:")
for i in order[:10]:
    print(f"  {cols[i]}: {importance[i]:.4f}")

# Save the plot as an image
out_dir = os.path.join(BASE, "06_reports")
os.makedirs(out_dir, exist_ok=True)
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150)
print("Saved plot to 06_reports/shap_summary.png")
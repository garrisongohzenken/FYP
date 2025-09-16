# baseline_linear_regression_manual.py
# Simple Linear Regression baseline WITHOUT sklearn Pipeline

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
# from utils.model_io import save_model

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/tech_salary_data_CLEANED.csv")

# Target variable
TARGET = "totalyearlycompensation"

# Drop rows with missing target
df = df.dropna(subset=[TARGET]).copy()

# -----------------------------
# 2. Select features
# -----------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# -----------------------------
# 3. One-Hot Encode categorical features manually
# -----------------------------
# Create dummy variables for categorical features
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# -----------------------------
# 4. Train-test split (75/25 split)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# Align train and test sets in case some categories are missing in test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# -----------------------------
# 5. Train Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Save model (timestamped filename)
# model_path = save_model(model, base_name="baseline_linear_regression", timestamp=True)
# print(f"Saved model to: {model_path}")

# -----------------------------
# 6. Predict & Evaluate
# -----------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("=== Baseline Linear Regression Metrics ===")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.6f}")

# -----------------------------
# 7. Plots
# -----------------------------
# Compute predictions and residuals for train and test
train_preds = model.predict(X_train)
test_preds = preds
resid_train = y_train - train_preds
resid_test = y_test - test_preds

# Create a single plot (left panel only from previous 1x2)
fig, ax = plt.subplots(figsize=(7, 3))

# Actual vs Predicted (with y=x reference)
ax.scatter(y_train, train_preds, alpha=0.3, label='Train', s=15)
ax.scatter(y_test, test_preds, alpha=0.6, label='Test', s=15)
min_y = float(min(y.min(), train_preds.min(), test_preds.min()))
max_y = float(max(y.max(), train_preds.max(), test_preds.max()))
ax.plot([min_y, max_y], [min_y, max_y], 'k--', linewidth=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
ax.legend()


plt.tight_layout()
plt.show()  


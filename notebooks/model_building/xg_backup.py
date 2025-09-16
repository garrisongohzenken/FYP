"""
XGBoost Regressor (no hyperparameter tuning)
- Same data prep as grid-search version
- Manual one-hot encoding, leakage-safe feature set
- Uses early stopping on a fixed validation split
- Evaluates on a held-out test set
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.model_io import save_model

from xgboost import XGBRegressor

# -----------------------------
# 1) Load data & target
# -----------------------------
CSV_PATH = "data/tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

# -----------------------------
# 2) Feature selection (avoid leakage)
# -----------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# -----------------------------
# 3) Manual one-hot encoding
# -----------------------------
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# -----------------------------
# 4) Train/Test split (final evaluation on test)
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)
# Align columns between train and test after OHE
X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

# -----------------------------
# 5) Internal validation split for early stopping
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# -----------------------------
# 6) Single XGBoost model (no tuning)
#    These are reasonable defaults; adjust as needed.
# -----------------------------
xgb = XGBRegressor(
    n_estimators=15000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="rmse",
    early_stopping_rounds=100,
    verbosity=0
)

# Fit with early stopping using the fixed validation set
xgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)]
)

# Save model (timestamped filename)
model_path = save_model(xgb, base_name="xgboost_no_tuning", timestamp=True)
print(f"Saved model to: {model_path}")

# -----------------------------
# 6a) Evaluate on the TRAIN set
# -----------------------------
train_preds = xgb.predict(X_train)
train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
train_r2 = r2_score(y_train, train_preds)

print("\n=== XGBoost (No Tuning) on TRAIN ===")
print(f"Best iteration: {getattr(xgb, 'best_iteration', 'N/A')}")
print(f"MAE:  {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R2:   {train_r2:.6f}")

# -----------------------------
# 6b) Evaluate on the VALIDATION set
# -----------------------------
val_preds = xgb.predict(X_valid)
val_mae = mean_absolute_error(y_valid, val_preds)
val_rmse = math.sqrt(mean_squared_error(y_valid, val_preds))
val_r2 = r2_score(y_valid, val_preds)

print("\n=== XGBoost (No Tuning) on VALIDATION ===")
print(f"Best iteration: {getattr(xgb, 'best_iteration', 'N/A')}")
print(f"MAE:  {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"R2:   {val_r2:.6f}")

# -----------------------------
# 7) Evaluate on the held-out TEST set
# -----------------------------
preds = xgb.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n=== XGBoost (No Tuning) on TEST ===")
print(f"Best iteration: {getattr(xgb, 'best_iteration', 'N/A')}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.6f}")

# -----------------------------
# 8) Feature importance (Top 20)
# -----------------------------
importances = pd.Series(xgb.feature_importances_, index=X_train_full.columns)
topk = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
topk.iloc[::-1].plot(kind="barh")
plt.title("XGBoost Feature Importance (Top 20)")
plt.xlabel("Gain-based Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# 9) Actual vs Predicted (train & test)
# -----------------------------
plt.figure(figsize=(6.5, 5))
plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
plt.scatter(y_test, preds, alpha=0.6, s=15, label="Test")
min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), preds.min()))
max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), preds.max()))
plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("XGBoost (No Tuning): Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

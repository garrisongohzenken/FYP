"""
XGBoost Regressor with RandomizedSearchCV (manual preprocessing, no Pipeline)
- Leakage-safe feature set and manual one-hot encoding
- Uses a fixed validation split for early stopping during search
- Refits best params with early stopping, evaluates on held-out test
- Plots Actual vs Predicted for Train & Test
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.model_io import save_model

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise SystemExit("Please install XGBoost:  pip install xgboost") from e

# -----------------------------
# 1) Load & target
# -----------------------------
CSV_PATH = "tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

# -----------------------------
# 2) Feature selection
# -----------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["title", "location", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# -----------------------------
# 3) Manual one-hot encoding
# -----------------------------
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# -----------------------------
# 4) Train / Test split (hold-out test for FINAL evaluation)
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)
X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

# -----------------------------
# 5) Internal validation split for early stopping (used during search and refit)
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# -----------------------------
# 6) RandomizedSearchCV over XGBoost
# -----------------------------
xgb_base = XGBRegressor(
    n_estimators=1200,        # upper bound; early stopping will cut it down
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="rmse",
    early_stopping_rounds=200,
    verbosity=0,
)

param_distributions = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 2, 4, 8, 16],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0, 0.5, 1, 3, 10, 20],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1],
    "gamma": [0, 0.1, 0.3, 1],
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=False,                     # we will refit manually with early stopping
)

# Fit with early stopping using fixed validation set
rand_search.fit(
    X_train, y_train,
    **{
        "eval_set": [(X_valid, y_valid)]   # XGBoost 2.x: eval_metric/early_stopping on estimator
    }
)

print("\n=== Randomized Search Results (XGBoost) ===")
print("Best params:", rand_search.best_params_)
print("Best CV score (neg MAE):", rand_search.best_score_)

# -----------------------------
# 7) Refit the best params with early stopping on (X_valid, y_valid)
# -----------------------------
best_params = rand_search.best_params_
xgb_best = XGBRegressor(
    n_estimators=2000,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="rmse",
    early_stopping_rounds=50,
    verbosity=0,
    **best_params,
)

xgb_best.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)]
)

# Save model (timestamped filename)
model_path = save_model(xgb_best, base_name="xgboost_randomizedsearchcv", timestamp=True)
print(f"Saved model to: {model_path}")

# -----------------------------
# 8) Evaluate on the held-out TEST set
# -----------------------------
preds = xgb_best.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n=== XGBoost (Best via RandomizedSearch) on TEST ===")
print(f"Best iteration: {xgb_best.best_iteration}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.6f}")

# -----------------------------
# 9) Feature importance (Top 20)
# -----------------------------
importances = pd.Series(xgb_best.feature_importances_, index=X_train_full.columns)
topk = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
topk.iloc[::-1].plot(kind="barh")
plt.title("XGBoost Feature Importance (Top 20)")
plt.xlabel("Gain-based Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# 10) Actual vs Predicted (train & test)
# -----------------------------
train_preds = xgb_best.predict(X_train)

plt.figure(figsize=(6.5, 5))
plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
plt.scatter(y_test, preds, alpha=0.6, s=15, label="Test")
min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), preds.min()))
max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), preds.max()))
plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("XGBoost (RandomizedSearch best): Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

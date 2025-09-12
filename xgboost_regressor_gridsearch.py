# xgboost_gridsearch_manual.py
# XGBoost Regressor with GridSearchCV (manual preprocessing, no Pipeline)
# - leakage-safe (drops basesalary/stockgrantvalue/bonus)
# - manual one-hot encoding
# - grid search over key hyperparameters (careful: can be large!)
# - uses early stopping during search (eval_set = fixed validation split)
# - refits best params with early stopping, then evaluates on test

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
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
# 3) Feature selection
# -----------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["title", "location", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# -----------------------------
# 4) Manual one-hot encoding
# -----------------------------
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# -----------------------------
# 5) Train / Test split (hold-out test for FINAL evaluation)
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)
X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

# -----------------------------
# 6) Create an internal validation split for early stopping
#    (used during randomized search and final refit)
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# -----------------------------
# 7) GridSearchCV over XGBoost
# -----------------------------
xgb_base = XGBRegressor(
    n_estimators=150,        # upper bound; early stopping will cut it down
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    # XGBoost 2.x: set eval_metric and early stopping on the estimator, not in fit()
    eval_metric="rmse",
    early_stopping_rounds=200,
    verbosity=0
)

param_grid = {
    "learning_rate": [0.01, 0.02, 0.03],
    "max_depth": [3, 4, 5],
    "min_child_weight": [2, 4, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_lambda": [0, 0.5, 2],
    "reg_alpha": [0, 0.1, 0.5],
    "gamma": [0, 0.1, 0.5],
}

# We still define a CV splitter, but early stopping will rely on (X_valid, y_valid).
# The scoring metric here is MAE (negated), but model training is controlled by early stopping on the fixed validation set.
cv = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=False,                        # we'll refit manually with early stopping
)

# Fit with early stopping using a fixed validation set
grid_search.fit(
    X_train, y_train,
    **{
        # XGBoost 2.x: only pass eval_set here; eval_metric/early_stopping are on the estimator
        "eval_set": [(X_valid, y_valid)]
    }
)

print("\n=== Grid Search Results (XGBoost) ===")
print("Best params:", grid_search.best_params_)
print("Best CV score (neg MAE):", grid_search.best_score_)

# -----------------------------
# 8) Refit the best params with early stopping on (X_valid, y_valid)
# -----------------------------
best_params = grid_search.best_params_
xgb_best = XGBRegressor(
    n_estimators=150,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="rmse",
    early_stopping_rounds=50,
    verbosity=0,
    **best_params
)

xgb_best.fit(
    X_train, y_train,
    # XGBoost 2.x: eval_metric/early_stopping configured on estimator
    eval_set=[(X_valid, y_valid)]
)

# Save model (timestamped filename)
model_path = save_model(xgboost_best := xgb_best, base_name="xgboost_gridsearchcv", timestamp=True)
print(f"Saved model to: {model_path}")

# -----------------------------
# 9) Evaluate on the held-out TEST set
# -----------------------------
preds = xgb_best.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n=== XGBoost (Best; early-stopped) on TEST ===")
print(f"Best iteration: {xgb_best.best_iteration}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.6f}")

# -----------------------------
# 10) Feature importance (Top 20)
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
# 11) Actual vs Predicted (train & test)
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
plt.title("XGBoost (GridSearch best): Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

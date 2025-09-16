# xgboost_catboost_pipeline.py

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from utils.model_io import save_model
from joblib import dump

# -----------------------------
# 1) Load data & target
# -----------------------------
CSV_PATH = "data/tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

# Log-transform target
y = np.log1p(df[TARGET].astype(float))

# Features
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats_high = [c for c in ["company", "title"] if c in df.columns]   # high cardinality → target encode
cat_feats_low = [c for c in ["country", "gender", "Race", "Education"] if c in df.columns]   # OHE

X = df[num_feats + cat_feats_high + cat_feats_low]

# -----------------------------
# 2) Encoding
# -----------------------------
# Target Encoding for high-cardinality
te = TargetEncoder(cols=cat_feats_high)
X_te = te.fit_transform(X[cat_feats_high], y)

# One-hot for low-cardinality
X_ohe = pd.get_dummies(X[cat_feats_low], drop_first=True)

# Combine
X_final = pd.concat([X[num_feats], X_te, X_ohe], axis=1)
feature_names = list(X_final.columns)

# -----------------------------
# 3) Train/Test Split
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_final, y, test_size=0.25, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# -----------------------------
# 4) Train XGBoost
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

xgb.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=200
)

# Save XGBoost model
xgb_path = save_model(xgb, base_name="xgboost", timestamp=True)
print(f"Saved XGBoost model to: {xgb_path}")

# -----------------------------
# 5) Train CatBoost
# -----------------------------
cat = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=5000,
    l2_leaf_reg=3,
    loss_function="RMSE",
    random_seed=42,
    verbose=200
)

cat.fit(
    X_train_full, y_train_full,
    eval_set=(X_test, y_test),
    cat_features=None  # already encoded
)

# Save CatBoost model
cat_path = save_model(cat, base_name="catboost", timestamp=True)
print(f"Saved CatBoost model to: {cat_path}")

# -----------------------------
# 6) Evaluation helper
# -----------------------------
def evaluate(model, Xtr, ytr, Xva, yva, Xte, yte, name="Model"):
    results = {}
    for split, X_, y_ in [("TRAIN", Xtr, ytr), ("VALID", Xva, yva), ("TEST", Xte, yte)]:
        preds_log = model.predict(X_)
        preds = np.expm1(preds_log)   # reverse log1p
        y_true = np.expm1(y_)         # reverse for metrics
        mae = mean_absolute_error(y_true, preds)
        rmse = math.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        print(f"\n=== {name} on {split} ===")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.6f}")
        results[split] = (mae, rmse, r2)
    return results

# Evaluate both models
res_xgb = evaluate(xgb, X_train, y_train, X_valid, y_valid, X_test, y_test, "XGBoost")
res_cat = evaluate(cat, X_train, y_train, X_valid, y_valid, X_test, y_test, "CatBoost")

# -----------------------------
# 7) Feature Importance (XGB)
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
# 8) Save encoder + column order sidecars (for dashboard alignment)
# -----------------------------
def _save_sidecars(model_path: str, encoder_obj, cols):
    base, _ = os.path.splitext(model_path)
    try:
        dump(encoder_obj, base + "_te.joblib")
    except Exception:
        pass
    try:
        with open(base + "_cols.json", "w", encoding="utf-8") as f:
            json.dump(list(map(str, cols)), f)
    except Exception:
        pass

try:
    _save_sidecars(xgb_path, te, feature_names)
except Exception:
    pass
try:
    _save_sidecars(cat_path, te, feature_names)
except Exception:
    pass

# catboost_pipeline.py

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

from catboost import CatBoostRegressor
from joblib import dump

# from utils.model_io import save_model

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
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.25, random_state=42
)

# -----------------------------
# 4) Train CatBoost
# -----------------------------
cat = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=5000,
    l2_leaf_reg=3,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
)

cat.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    cat_features=None  # already encoded
)

# Save CatBoost model
# cat_path = save_model(cat, base_name="catboost", timestamp=True)
# print(f"Saved CatBoost model to: {cat_path}")

# -----------------------------
# 5) Evaluation helper
# -----------------------------
def evaluate(model, Xtr, ytr, Xte, yte, name="Model"):
    results = {}
    for split, X_, y_ in [("TRAIN", Xtr, ytr), ("TEST", Xte, yte)]:
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

# Evaluate CatBoost
res_cat = evaluate(cat, X_train, y_train, X_test, y_test, "CatBoost")

# -----------------------------
# 6) Feature Importance (CatBoost)
# -----------------------------
importances = pd.Series(cat.feature_importances_, index=X_train.columns)
topk = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
topk.iloc[::-1].plot(kind="barh")
plt.title("CatBoost Feature Importance (Top 20)")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Actual vs Predicted Plot (Train/Test)
# -----------------------------
# Get predictions in original scale
train_preds = np.expm1(cat.predict(X_train))
test_preds = np.expm1(cat.predict(X_test))
y_train_orig = np.expm1(y_train)
y_test_orig = np.expm1(y_test)

plt.figure(figsize=(6.5, 5))
plt.scatter(y_train_orig, train_preds, alpha=0.3, s=15, label="Train")
plt.scatter(y_test_orig, test_preds, alpha=0.6, s=15, label="Test")
min_y = float(min(y_train_orig.min(), y_test_orig.min(), train_preds.min(), test_preds.min()))
max_y = float(max(y_train_orig.max(), y_test_orig.max(), train_preds.max(), test_preds.max()))
plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("CatBoost: Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Save encoder + column order sidecars (for dashboard alignment)
# -----------------------------
# def _save_sidecars(model_path: str, encoder_obj, cols):
#     base, _ = os.path.splitext(model_path)
#     try:
#         dump(encoder_obj, base + "_te.joblib")
#     except Exception:
#         pass
#     try:
#         with open(base + "_cols.json", "w", encoding="utf-8") as f:
#             json.dump(list(map(str, cols)), f)
#     except Exception:
#         pass

# try:
#     _save_sidecars(cat_path, te, feature_names)
# except Exception:
#     pass

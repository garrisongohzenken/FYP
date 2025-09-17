"""
Random Forest Regressor (no hyperparameter tuning)
- Same data prep as tuned versions
- Manual one-hot encoding, leakage-safe feature set
- Trains a single RandomForestRegressor with sensible defaults
- Evaluates on a held-out test set
"""

import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from utils.model_io import save_model

# --------------------------------
# 1) Load dataset
# --------------------------------
CSV_PATH = "data/tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

# --------------------------------
# 2) Select features
# --------------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# --------------------------------
# 3) One-hot encode categoricals (manual)
# --------------------------------
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# --------------------------------
# 4) Train/test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# Align columns in case test lacks some training categories
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --------------------------------
# 5) Single RandomForest mode
# --------------------------------
rf = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=20,
    min_samples_leaf=1,
    max_features=0.1,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train)

# Save model (timestamped filename)
# model_path = save_model(rf, base_name="random_forest_no_tuning", timestamp=True)
# print(f"Saved model to: {model_path}")

# --------------------------------
# 6) Evaluate on train set
# --------------------------------
train_preds = rf.predict(X_train)
train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = math.sqrt(mean_squared_error(y_train, train_preds))
train_r2 = r2_score(y_train, train_preds)

print("\n=== Random Forest on Train ===")
print(f"MAE:  {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R2:   {train_r2:.6f}")

# --------------------------------
# 7) Evaluate on test set
# --------------------------------
test_preds = rf.predict(X_test)
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
test_r2 = r2_score(y_test, test_preds)

print("\n=== Random Forest on Test ===")
print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R2:   {test_r2:.6f}")

# --------------------------------
# 8) Feature importance (Top 20)
# --------------------------------
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
topk = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
topk.iloc[::-1].plot(kind="barh")
plt.title("Random Forest Feature Importance (Top 20)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# --------------------------------
# 9) Actual vs Predicted (train & test)
# --------------------------------
plt.figure(figsize=(6.5, 5))
plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
plt.scatter(y_test, test_preds, alpha=0.6, s=15, label="Test")
min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), test_preds.min()))
max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), test_preds.max()))
plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest: Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

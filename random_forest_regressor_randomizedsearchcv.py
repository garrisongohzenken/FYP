# random_forest_regressor_randomizedsearch.py
# Random Forest with RandomizedSearchCV (manual preprocessing, no Pipeline)

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.model_io import save_model

# --------------------------------
# 1) Load dataset
# --------------------------------
CSV_PATH = "tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

# --------------------------------
# 3) Select features
# --------------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["title", "location", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# --------------------------------
# 4) One-hot encode categoricals (manual)
# --------------------------------
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# --------------------------------
# 5) Train/test split (seed=42 for comparability)
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# Align columns in case test lacks some training categories
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --------------------------------
# 6) RandomizedSearch over RandomForest hyperparameters
#    (uses lists so there is no SciPy dependency)
# --------------------------------
rf = RandomForestRegressor(n_jobs=-1, random_state=42)

param_distributions = {
    "n_estimators": [200, 300, 400, 500, 600, 700],
    "max_depth": [None, 10, 16, 24, 32, 40],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
    "bootstrap": [True],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

rand = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=40,                         # increase for a broader search
    scoring="neg_mean_absolute_error", # MAE in dollars
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rand.fit(X_train, y_train)

print("\n=== Randomized Search Results ===")
print("Best params:", rand.best_params_)
print("CV best (negative MAE):", rand.best_score_)

best_model = rand.best_estimator_

# Save model (timestamped filename)
model_path = save_model(best_model, base_name="random_forest_randomizedsearchcv", timestamp=True)
print(f"Saved model to: {model_path}")

# --------------------------------
# 7) Evaluate on test set
# --------------------------------
preds = best_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n=== Random Forest (Best Model on Test) ===")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.6f}")

# --------------------------------
# 8) Feature importance (Top 20)
# --------------------------------
importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
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
train_preds = best_model.predict(X_train)

plt.figure(figsize=(6.5, 5))
plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
plt.scatter(y_test, preds, alpha=0.6, s=15, label="Test")
min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), preds.min()))
max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), preds.max()))
plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest (Best via RandomizedSearch): Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

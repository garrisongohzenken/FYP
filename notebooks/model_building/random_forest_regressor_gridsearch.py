# random_forest_regressor_gridsearch.py
# Random Forest with hyperparameter tuning (manual preprocessing, no Pipeline)

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.model_io import save_model

# --------------------------------
# 1) Load dataset
# --------------------------------
CSV_PATH = "data/tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET]).copy()

# --------------------------------
# 2) Select features
# --------------------------------
num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_feats = [c for c in ["title", "location", "gender", "Race", "Education"] if c in df.columns]

X = df[num_feats + cat_feats]
y = df[TARGET].astype(float)

# --------------------------------
# 3) One-hot encode categoricals (manual, no Pipeline)
# --------------------------------
X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

# --------------------------------
# 4) Train/test split (seed=42 for comparability with your baseline) (75/25 split)
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

# Align columns in case test lacks some training categories
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --------------------------------
# 5) Grid Search over RandomForest hyperparameters
#    (balanced grid for speed & quality; expand if you want deeper search)
# --------------------------------
rf = RandomForestRegressor(n_jobs=-1, random_state=42)

param_grid = {
    "n_estimators": [75, 100, 200],      # more trees => better stability (slower)
    "max_depth": [None, 12, 24],      # None lets nodes expand fully
    "min_samples_split": [5, 10, 20],      # larger -> smoother trees
    "min_samples_leaf": [2, 4, 6],        # larger -> smoother trees
    "max_features": ["sqrt", "log2", 0.5, 1.0],  # fraction or method
    "bootstrap": [True],                  # typical for RF
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",    # MAE is intuitive for $ errors
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\n=== Grid Search Results ===")
print("Best params:", grid.best_params_)
print("CV best (negative MAE):", grid.best_score_)

best_model = grid.best_estimator_

# Save model (timestamped filename)
model_path = save_model(best_model, base_name="random_forest_gridsearchcv", timestamp=True)
print(f"Saved model to: {model_path}")

# --------------------------------
# 6) Evaluate on test set
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
# 7) Feature importance (Top 20)
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
plt.title("Random Forest: Actual vs Predicted (Train/Test)")
plt.legend()
plt.tight_layout()
plt.show()

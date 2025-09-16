"""
XGBoost Regressor — Compact GridSearchCV Tuning

- Matches xg_backup.py preprocessing/splits for fair comparison
- Uses a small, targeted grid to keep compute light (~64 fits with 3-fold CV)
- Evaluates on a held-out test set and saves the best model
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from utils.model_io import save_model


def main():
    # 1) Load (same as xg_backup)
    CSV_PATH = "data/tech_salary_data_CLEANED.csv"
    TARGET = "totalyearlycompensation"
    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

    # 2) Feature selection (same pattern)
    num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
    cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

    X = pd.get_dummies(df[num_feats + cat_feats], columns=cat_feats, drop_first=True)
    y = df[TARGET].astype(float)

    # 3) Train/Test split (and align test columns)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

    # Internal validation is handled by CV (GridSearchCV)

    # 4) Model + compact grid (resource-light)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="rmse",
        verbosity=0,
    )

    # ~16–64 combos depending on toggles below (cv=3 -> 48–192 fits)
    param_grid = {
        "n_estimators": [10000],
        "learning_rate": [0.001],
        "max_depth": [8],
        "min_child_weight": [2],
        # Keep these narrow to limit fits; widen if you have headroom
        "subsample": [0.75],
        "colsample_bytree": [0.6],
        # Regularization kept fixed initially; uncomment to expand search
        "reg_lambda": [1],
        "reg_alpha": [9.5],
        "gamma": [0],
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",  # optimize RMSE directly
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train_full, y_train_full)
    best_xgb = grid.best_estimator_

    print("\nBest params:", grid.best_params_)
    print("Best CV RMSE:", -grid.best_score_)

    # 5) Evaluate on TEST
    preds = best_xgb.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== XGBoost (GridSearchCV) on TEST ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 6) Save + importance plot
    path = save_model(best_xgb, base_name="xgboost_tuned_grid", timestamp=True)
    print(f"Saved model to: {path}")

    importances = pd.Series(best_xgb.feature_importances_, index=X_train_full.columns)
    topk = importances.sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    topk.iloc[::-1].plot(kind="barh")
    plt.title("XGBoost Feature Importance (Top 20)")
    plt.xlabel("Gain-based Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


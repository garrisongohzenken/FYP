"""
LightGBM Regressor — Baseline (no tuning)

- Manual one-hot encoding to match existing scripts
- Train/valid split for early stopping, then evaluate on a held-out test set
- Saves the trained model and shows quick diagnostics
"""

import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.model_io import save_model


def main():
    # 1) Load dataset
    CSV_PATH = "data/tech_salary_data_CLEANED.csv"
    TARGET = "totalyearlycompensation"
    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

    # 2) Select features
    num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
    cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

    X = df[num_feats + cat_feats]
    y = df[TARGET].astype(float)

    # Manual OHE to match the project style
    X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)

    # LightGBM cannot handle certain JSON-unsafe characters in feature names.
    # Sanitize column names to avoid characters like quotes, brackets, braces, colon, comma, spaces, etc.
    def _sanitize_names(cols: pd.Index) -> pd.Index:
        # Replace unsafe characters with underscore
        safe = cols.str.replace(r'[\"\[\]\{\}\:\,]', '_', regex=True).str.replace(r"\s+", "_", regex=True)
        # Ensure uniqueness after replacement
        seen = {}
        unique = []
        for c in safe:
            base = c
            i = 1
            while c in seen:
                i += 1
                c = f"{base}__{i}"
            seen[c] = True
            unique.append(c)
        return pd.Index(unique)

    X_encoded.columns = _sanitize_names(X_encoded.columns)

    # 3) Train/Test split (final evaluation on test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42
    )
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

    # Internal validation split for early stopping
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42
    )

    # 4) LightGBM baseline (no tuning)
    lgbm = LGBMRegressor(
        n_estimators=5000,        # upper bound; early stopping finds best_iteration
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        random_state=42,
        objective="regression",
    )

    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # 5) Evaluate on TEST
    preds = lgbm.predict(X_test, num_iteration=lgbm.best_iteration_)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== LightGBM Baseline on TEST ===")
    print(f"Best iteration: {getattr(lgbm, 'best_iteration_', 'N/A')}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 6) Save and quick plots
    path = save_model(lgbm, base_name="lgbm_baseline", timestamp=True)
    print(f"Saved model to: {path}")

    importances = pd.Series(lgbm.feature_importances_, index=X_train_full.columns)
    topk = importances.sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    topk.iloc[::-1].plot(kind="barh")
    plt.title("LightGBM Feature Importance (Top 20)")
    plt.xlabel("Split Gain Importance")
    plt.tight_layout()
    plt.show()

    # Actual vs Predicted
    train_preds = lgbm.predict(X_train, num_iteration=lgbm.best_iteration_)
    plt.figure(figsize=(6.5, 5))
    plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
    plt.scatter(y_test, preds, alpha=0.6, s=15, label="Test")
    min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), preds.min()))
    max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), preds.max()))
    plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("LightGBM Baseline: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

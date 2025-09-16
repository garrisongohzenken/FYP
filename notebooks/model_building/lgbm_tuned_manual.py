"""
LightGBM Regressor — Manual Randomized Tuning with Early Stopping

- Matches lgbm_baseline.py preprocessing (manual one-hot + column sanitization)
- Randomly samples compact hyperparameters for N trials (default 30)
- Uses early stopping per trial on a fixed validation split
- Selects the best by validation RMSE, evaluates on test, saves the model
"""

import argparse
import math
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.model_io import save_model


def sanitize_columns(cols: pd.Index) -> pd.Index:
    """Replace JSON-unsafe chars and enforce uniqueness (matches baseline)."""
    safe = (
        cols.str.replace(r'[\\"\[\]\{\}\:\,]', '_', regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    seen = {}
    out = []
    for c in safe:
        base = c
        i = 1
        while c in seen:
            i += 1
            c = f"{base}__{i}"
        seen[c] = True
        out.append(c)
    return pd.Index(out)


def sample_params(rng: random.Random) -> Dict[str, Any]:
    """Sample a LightGBM hyperparameter configuration."""
    return {
        # tree structure
        "num_leaves": rng.choice([31, 63, 127, 255]),
        "max_depth": rng.choice([-1, 6, 8, 10, 12]),
        "min_child_samples": rng.choice([5, 10, 20, 40]),
        # learning dynamics
        "learning_rate": rng.choice([0.01, 0.02, 0.03, 0.05]),
        # sampling
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.7, 0.8, 0.9, 1.0]),
        # regularization
        "reg_lambda": rng.choice([0.0, 0.5, 1.0, 2.0, 5.0, 10.0]),
        "reg_alpha": rng.choice([0.0, 0.001, 0.01, 0.1, 1.0]),
        "min_split_gain": rng.choice([0.0, 0.1, 0.5, 1.0]),
    }


def main(trials: int, seed: int) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    # 1) Load dataset
    CSV_PATH = "data/tech_salary_data_CLEANED.csv"
    TARGET = "totalyearlycompensation"
    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

    # 2) Select features (match baseline)
    num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
    cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

    X = df[num_feats + cat_feats]
    y = df[TARGET].astype(float)

    # Manual OHE to match project style
    X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)
    X_encoded.columns = sanitize_columns(X_encoded.columns)

    # 3) Train/Test split (final evaluation on test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42
    )
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

    # Internal validation split for early stopping
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42
    )

    best = {"rmse": float("inf"), "params": None, "model": None}

    for t in range(1, trials + 1):
        params = sample_params(rng)
        model = LGBMRegressor(
            n_estimators=8000,        # upper bound, early stopping chooses best
            objective="regression",
            boosting_type="gbdt",
            n_jobs=-1,
            random_state=seed,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        pred_val = model.predict(X_valid, num_iteration=model.best_iteration_)
        rmse_val = math.sqrt(mean_squared_error(y_valid, pred_val))

        if rmse_val < best["rmse"]:
            best = {"rmse": rmse_val, "params": params, "model": model}

        if t % max(1, trials // 5) == 0 or t == trials:
            print(f"Trial {t}/{trials} — val RMSE: {rmse_val:.4f} | best: {best['rmse']:.4f}")

    # 4) Test evaluation with best model
    best_model: LGBMRegressor = best["model"]
    print("\nBest params:", best["params"])
    print("Best val RMSE:", best["rmse"])
    print("Best iteration:", getattr(best_model, "best_iteration_", "N/A"))

    preds_test = best_model.predict(X_test, num_iteration=best_model.best_iteration_)
    mae = mean_absolute_error(y_test, preds_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds_test))
    r2 = r2_score(y_test, preds_test)

    print("\n=== LightGBM (Manual Randomized Tuning) on TEST ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 5) Save model
    path = save_model(best_model, base_name="lgbm_tuned_manual", timestamp=True)
    print(f"Saved model to: {path}")

    # 6) Feature importance plot
    importances = pd.Series(best_model.feature_importances_, index=X_train_full.columns)
    topk = importances.sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    topk.iloc[::-1].plot(kind="barh")
    plt.title("LightGBM Feature Importance (Top 20)")
    plt.xlabel("Split Gain Importance")
    plt.tight_layout()
    plt.show()

    # 7) Actual vs Predicted
    train_preds = best_model.predict(X_train, num_iteration=best_model.best_iteration_)
    plt.figure(figsize=(6.5, 5))
    plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
    plt.scatter(y_test, preds_test, alpha=0.6, s=15, label="Test")
    min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), preds_test.min()))
    max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), preds_test.max()))
    plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("LightGBM (Manual Tuning): Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM manual randomized tuning with early stopping")
    parser.add_argument("--trials", type=int, default=30, help="Number of random trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(trials=args.trials, seed=args.seed)


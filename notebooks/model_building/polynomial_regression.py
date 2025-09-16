"""
Polynomial Regression — Baseline (no tuning)

- Uses a clean sklearn Pipeline with ColumnTransformer
- Applies PolynomialFeatures to numeric columns only (degree=2)
- One-hot encodes categoricals with handle_unknown='ignore'
- Evaluates on a held-out test set and saves the model
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.model_io import save_model


def main():
    # 1) Load dataset
    CSV_PATH = "data/tech_salary_data_CLEANED.csv"
    TARGET = "totalyearlycompensation"
    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

    # 2) Columns
    num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
    cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

    X = df[num_feats + cat_feats]
    y = df[TARGET].astype(float)

    # 3) Preprocessing: degree-2 poly on numeric, OHE on categoricals
    numeric_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler(with_mean=False)),  # sparse-friendly
    ])

    categorical_pipeline = OneHotEncoder(handle_unknown="ignore", drop="first")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_feats),
            ("cat", categorical_pipeline, cat_feats),
        ],
        sparse_threshold=0.3,
    )

    model = Pipeline([
        ("prep", preprocess),
        ("reg", LinearRegression()),
    ])

    # 4) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5) Fit and evaluate
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== Polynomial Regression (degree=2) on TEST ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 6) Save model
    path = save_model(model, base_name="polynomial_regression_baseline", timestamp=True)
    print(f"Saved model to: {path}")

    # 7) Actual vs Predicted
    train_preds = model.predict(X_train)
    plt.figure(figsize=(6.5, 5))
    plt.scatter(y_train, train_preds, alpha=0.3, s=15, label="Train")
    plt.scatter(y_test, preds, alpha=0.6, s=15, label="Test")
    min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), preds.min()))
    max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), preds.max()))
    plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Polynomial Regression Baseline: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


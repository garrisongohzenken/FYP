"""
XGBoost Regressor — Efficient Hyperparameter Tuning (HalvingRandomSearchCV)

- Reuses the same manual one-hot encoding approach as existing scripts
- Successive halving prunes weak configs early by growing n_estimators
- Uses a log-target transform to stabilize heavy-tailed compensation
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV

from xgboost import XGBRegressor
from utils.model_io import save_model


def collapse_rare(series: pd.Series, top_k: int = 50) -> pd.Series:
    """Collapse infrequent categories to 'Other' to keep dummies manageable."""
    vc = series.value_counts(dropna=False)
    keep = set(vc.nlargest(top_k).index)
    return series.astype("string").where(series.isin(keep), "Other")


def main():
    # 1) Load
    CSV_PATH = "data/tech_salary_data_CLEANED.csv"
    TARGET = "totalyearlycompensation"
    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET]).copy()

    # 2) Features
    num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
    cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

    X = df[num_feats + cat_feats]
    y = df[TARGET].astype(float)

    for c in cat_feats:
        X[c] = collapse_rare(X[c])

    # One-hot encode
    X = pd.get_dummies(X, columns=cat_feats, drop_first=True)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 4) HalvingRandomSearchCV with log-target transform
    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="rmse",
        # n_estimators is managed by halving via the resource parameter
    )

    model = TransformedTargetRegressor(
        regressor=base_xgb, func=np.log1p, inverse_func=np.expm1
    )

    param_distributions = {
        # Depth/leaf complexity
        "regressor__max_depth": [4, 6, 8, 10],
        "regressor__min_child_weight": [1, 2, 5, 10],
        # Learning dynamics
        "regressor__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        "regressor__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        # Regularization
        "regressor__reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
        "regressor__reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0, 5.0],
        "regressor__gamma": [0.0, 0.1, 0.5, 1.0],
    }

    search = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        factor=3,
        resource="regressor__n_estimators",  # progressively increase trees
        min_resources=200,
        max_resources=3000,
        aggressive_elimination=True,
        random_state=42,
        n_candidates="exhaust",
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    # 5) Evaluate on test
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== XGBoost (HalvingRandomSearchCV) on Test ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 6) Save + importance plot
    path = save_model(best_model, base_name="xgboost_tuned_halving", timestamp=True)
    print(f"Saved model to: {path}")

    # Extract underlying XGB for feature importances
    fitted_xgb = getattr(best_model, "regressor_", None)
    if fitted_xgb is None:
        fitted_xgb = best_model
    importances = pd.Series(fitted_xgb.feature_importances_, index=X_train.columns)
    topk = importances.sort_values(ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    topk.iloc[::-1].plot(kind="barh")
    plt.title("XGBoost Feature Importance (Top 20)")
    plt.xlabel("Gain-based Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


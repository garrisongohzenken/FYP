"""
Random Forest Regressor — HalvingGridSearchCV (Efficient)

- Reuses rf_backup's data prep (manual one-hot)
- Successive halving search to reduce total fits drastically
- Uses log-target transform to stabilize heavy-tailed salaries
- Evaluates on a held-out test set and saves the best model
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

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

    # 2) Features (same pattern as rf_backup)
    num_feats = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
    cat_feats = [c for c in ["company", "title", "country", "gender", "Race", "Education"] if c in df.columns]

    X = df[num_feats + cat_feats]
    y = df[TARGET].astype(float)

    for c in cat_feats:
        X[c] = collapse_rare(X[c])

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=cat_feats, drop_first=True)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 4) HalvingGridSearchCV with log-target transform
    base_rf = RandomForestRegressor(bootstrap=True, n_jobs=-1, random_state=42)
    # Fit on log1p target, predict in original space via expm1
    model = TransformedTargetRegressor(
        regressor=base_rf, func=np.log1p, inverse_func=np.expm1
    )

    # Grid without n_estimators (handled as the resource for halving)
    param_grid = {
        "regressor__max_depth": [None, 16, 24, 32],
        "regressor__min_samples_split": [2, 5, 10, 20],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    grid = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
        factor=3,
        resource="regressor__n_estimators",  # grow number of trees
        min_resources=100,
        max_resources=1200,
        aggressive_elimination=True,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\nBest params:", grid.best_params_)
    print("Best CV MAE:", -grid.best_score_)

    # 5) Test evaluation (best_model already predicts on original scale)
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== Random Forest (GridSearchCV) on Test ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 6) Save + quick importance plot
    model_path = save_model(best_model, base_name="random_forest_tuned_grid", timestamp=True)
    print(f"Saved model to: {model_path}")

    # Extract underlying RF from TransformedTargetRegressor for importances
    fitted_rf = getattr(best_model, "regressor_", None)
    if fitted_rf is None:
        # Fallback: try direct attribute (if not wrapped)
        fitted_rf = best_model
    importances = pd.Series(fitted_rf.feature_importances_, index=X_train.columns)
    topk = importances.sort_values(ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    topk.iloc[::-1].plot(kind="barh")
    plt.title("Random Forest Feature Importance (Top 20)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

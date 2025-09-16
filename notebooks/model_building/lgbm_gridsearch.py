"""
LightGBM Regressor — Compact GridSearchCV Tuning

- Matches lgbm_baseline preprocessing (manual one-hot + column sanitization)
- Compact grid to keep fits reasonable with CV=3
- Evaluates best model on held-out test set and saves it
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb  # noqa: F401  # kept for parity/logging
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

from utils.model_io import save_model


def sanitize_columns(cols: pd.Index) -> pd.Index:
    """Replace JSON-unsafe chars and enforce uniqueness (same as baseline)."""
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

    # Collapse rare categories to cut feature dimensionality
    for c in cat_feats:
        if c in X.columns:
            top = X[c].value_counts().nlargest(50).index
            X[c] = X[c].where(X[c].isin(top), "Other")

    X_encoded = pd.get_dummies(X, columns=cat_feats, drop_first=True)
    X_encoded.columns = sanitize_columns(X_encoded.columns)

    # 3) Train/Test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42
    )
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)
    # Create a fixed validation split for early stopping during HalvingGridSearchCV
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    # Combine for PredefinedSplit so each fit uses the same validation fold
    X_cv = pd.concat([X_tr, X_val], axis=0)
    y_cv = pd.concat([y_tr, y_val], axis=0)
    test_fold = np.r_[np.full(len(X_tr), -1), np.zeros(len(X_val))]
    ps = PredefinedSplit(test_fold=test_fold)

    # 4) Estimator wrapped to train on log1p(target) and predict on original scale
    base = LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_jobs=-1,
        random_state=42,
        force_col_wise=True
    )
    model = TransformedTargetRegressor(
        regressor=base, func=np.log1p, inverse_func=np.expm1
    )

    # Prefix params with 'regressor__' to target the wrapped LGBM estimator
    # Tighter grid to reduce total candidates (fast but effective)
    param_grid = {
        # n_estimators is managed as the halving resource; do not include here
        "regressor__learning_rate": [0.01, 0.02],
        "regressor__num_leaves": [127, 255],
        "regressor__max_depth": [-1, 12],
        "regressor__min_child_samples": [5, 10],
        "regressor__subsample": [0.9, 1.0],
        "regressor__colsample_bytree": [0.8, 1.0],
        "regressor__min_split_gain": [0.0],
    }

    grid = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=ps,
        n_jobs=-1,
        verbose=1,
        factor=4,
        resource="regressor__n_estimators",
        min_resources=1200,
        max_resources=6000,
        aggressive_elimination=True,
    )

    # Early stopping on the fixed validation fold; note y must be log-transformed
    fit_params = dict(
        eval_set=[(X_val, np.log1p(y_val))],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    grid.fit(X_cv, y_cv, **fit_params)
    best_model = grid.best_estimator_

    print("\nBest params:", grid.best_params_)
    print("Best CV RMSE:", -grid.best_score_)

    # 5) Evaluate on TEST
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== LightGBM (HalvingGridSearchCV) on TEST ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.6f}")

    # 5a) Regression plot (Test): predicted vs actual with fit line
    plt.figure(figsize=(6.5, 5))
    sns.regplot(
        x=y_test,
        y=preds,
        scatter_kws={"alpha": 0.35, "s": 15},
        line_kws={"color": "crimson", "lw": 1.5},
        lowess=False,
    )
    min_y = float(min(y_test.min(), preds.min()))
    max_y = float(max(y_test.max(), preds.max()))
    plt.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
    plt.xlabel("Actual (Test)")
    plt.ylabel("Predicted (Test)")
    plt.title("LightGBM (HalvingGridSearchCV): Regression Plot (Test)")
    plt.tight_layout()
    plt.show()

    # 6) Save + importance plot
    path = save_model(best_model, base_name="lgbm_tuned_grid", timestamp=True)
    print(f"Saved model to: {path}")

    # Get feature importances from underlying fitted LGBM model
    fitted_lgbm = getattr(best_model, "regressor_", best_model)
    importances = pd.Series(fitted_lgbm.feature_importances_, index=X_train_full.columns)
    topk = importances.sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    topk.iloc[::-1].plot(kind="barh")
    plt.title("LightGBM Feature Importance (Top 20)")
    plt.xlabel("Split Gain Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

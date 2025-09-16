# filename: salary_ensemble_oof.py
# Purpose: Stronger generalization via out-of-fold (OOF) stacking (XGBoost + CatBoost -> Ridge)

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from category_encoders import TargetEncoder

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# -----------------------------
# 0) Config
# -----------------------------
CSV_PATH = "data/tech_salary_data_CLEANED.csv"  # change if needed
TARGET = "totalyearlycompensation"
RANDOM_STATE = 42
N_FOLDS = 5
EARLY_STOP_ROUNDS = 300

# -----------------------------
# 1) Load
# -----------------------------
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(float)

# -----------------------------
# 2) Base feature lists
# -----------------------------
num_base = [c for c in ["yearsofexperience", "yearsatcompany"] if c in df.columns]
cat_high = [c for c in ["company", "title"] if c in df.columns]  # target-encode
cat_low  = [c for c in ["country", "gender", "Race", "Education"] if c in df.columns]  # OHE via pd.get_dummies

# -----------------------------
# 3) Light feature engineering
# -----------------------------
def region_from_country(country: str) -> str:
    if not isinstance(country, str):
        return "Other"
    c = country.lower()
    na = {"united states","usa","canada","mexico"}
    eu = {"united kingdom","germany","france","ireland","netherlands","sweden","spain","italy","switzerland","poland"}
    apac = {"singapore","malaysia","india","china","japan","south korea","australia","new zealand","vietnam","indonesia","thailand","philippines","taiwan","hong kong (sar)"}
    if c in na: return "NA"
    if c in eu: return "EU"
    if c in apac: return "APAC"
    return "Other"

if "country" in df.columns and "region" not in df.columns:
    df["region"] = df["country"].apply(region_from_country)
    if "region" not in cat_low: cat_low.append("region")

def seniority_from_title(title: str) -> str:
    if not isinstance(title, str):
        return "Unknown"
    t = title.lower()
    if any(k in t for k in ["director","head","principal","lead staff","vp","vice president"]): return "Director+"
    if any(k in t for k in ["manager","mgr","lead"]): return "Manager"
    if any(k in t for k in ["sr","senior"," ii"," iii"]): return "Senior"
    if any(k in t for k in ["intern","junior","jr","trainee","apprentice"]): return "Junior"
    return "Mid"

if "title" in df.columns and "seniority" not in df.columns:
    df["seniority"] = df["title"].apply(seniority_from_title)
    if "seniority" not in cat_low: cat_low.append("seniority")

def is_bigtech(company: str) -> int:
    if not isinstance(company, str):
        return 0
    c = company.lower()
    big = ["google","alphabet","amazon","apple","meta","facebook","netflix","microsoft"]
    return int(any(b in c for b in big))

if "company" in df.columns and "is_bigtech" not in df.columns:
    df["is_bigtech"] = df["company"].apply(is_bigtech)
    if "is_bigtech" not in num_base: num_base.append("is_bigtech")

# log/sqrt transforms
if "yearsofexperience" in df.columns:
    df["log_experience"] = np.log1p(df["yearsofexperience"].clip(lower=0))
    df["sqrt_experience"] = np.sqrt(df["yearsofexperience"].clip(lower=0))
if "yearsatcompany" in df.columns:
    df["log_yearsatcompany"] = np.log1p(df["yearsatcompany"].clip(lower=0))

# interactions
if set(["yearsofexperience","yearsatcompany"]).issubset(df.columns):
    denom = df["yearsofexperience"].replace(0, np.nan)
    df["loyalty_ratio"] = (df["yearsatcompany"]/denom).fillna(0).clip(0,1.0)

num_final = [c for c in [
    "is_bigtech",
    "yearsofexperience","yearsatcompany",
    "log_experience","sqrt_experience","log_yearsatcompany",
    "loyalty_ratio"
] if c in df.columns]

# -----------------------------
# 4) Assemble design matrix
# -----------------------------
X_all = pd.DataFrame(index=df.index)
X_all[num_final] = df[num_final]
X_all[cat_high + cat_low] = df[cat_high + cat_low]
y_all = np.log1p(df[TARGET].values)  # log target

# -----------------------------
# 5) Train/Test split once (hold TEST untouched)
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=RANDOM_STATE
)

# One-hot encode low-cardinality **only after** TE is added in each fold.
low_cols = [c for c in cat_low if c in X_train_full.columns]

# -----------------------------
# 6) CV setup (Stratify by target bins for balance)
# -----------------------------
y_bins = pd.qcut(y_train_full, q=min(10, len(y_train_full)//50), duplicates="drop").codes
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Prepare OOF containers
oof_pred_xgb = np.zeros_like(y_train_full, dtype=float)
oof_pred_cat = np.zeros_like(y_train_full, dtype=float)
test_pred_xgb_folds = []
test_pred_cat_folds = []

# -----------------------------
# 7) Loop folds
# -----------------------------
X_train_full = X_train_full.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train_full = y_train_full.astype(float)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_full, y_bins), 1):
    X_tr, X_va = X_train_full.iloc[tr_idx].copy(), X_train_full.iloc[va_idx].copy()
    y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]

    # --- Target Encoding for high-cardinality (fit on TR only)
    te = TargetEncoder(cols=[c for c in cat_high if c in X_tr.columns],
                       smoothing=0.3, min_samples_leaf=20)
    if len(te.cols_) if hasattr(te, "cols_") else len(te.cols) > 0:
        te.fit(X_tr[cat_high], y_tr)
        for split_df in [X_tr, X_va]:
            te_out = te.transform(split_df[cat_high])
            te_out.columns = [f"TE_{c}" for c in te_out.columns]
            split_df.drop(columns=cat_high, inplace=True)
            split_df[te_out.columns] = te_out.values
        # For TEST: transform with the same TE
        X_te = X_test.copy()
        te_out_test = te.transform(X_te[cat_high])
        te_out_test.columns = [f"TE_{c}" for c in te_out_test.columns]
        X_te.drop(columns=cat_high, inplace=True)
        X_te[te_out_test.columns] = te_out_test.values
    else:
        X_te = X_test.copy()

    # --- OHE for low-cardinality (fit structure on TR)
    tr_ohe = pd.get_dummies(X_tr, columns=low_cols, drop_first=True)
    va_ohe = pd.get_dummies(X_va, columns=low_cols, drop_first=True)
    te_ohe = pd.get_dummies(X_te, columns=low_cols, drop_first=True)

    # align columns
    va_ohe = va_ohe.reindex(columns=tr_ohe.columns, fill_value=0)
    te_ohe = te_ohe.reindex(columns=tr_ohe.columns, fill_value=0)

    # --- XGBoost with early stopping
    xgb = XGBRegressor(
        n_estimators=20000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=1.0,
        min_child_weight=3,
        gamma=0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbosity=0
    )
    xgb.fit(tr_ohe, y_tr, eval_set=[(va_ohe, y_va)], verbose=False)

    # --- CatBoost with early stopping
    cat = CatBoostRegressor(
        depth=6,
        learning_rate=0.03,
        iterations=20000,
        l2_leaf_reg=14,
        subsample=0.8,
        rsm=0.8,                 # column sampling
        random_strength=0.3,     # stronger regularization
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False
    )
    cat.fit(tr_ohe, y_tr, eval_set=(va_ohe, y_va), use_best_model=True, verbose=False)

    # --- OOF predictions
    oof_pred_xgb[va_idx] = xgb.predict(va_ohe)
    oof_pred_cat[va_idx] = cat.predict(va_ohe)

    # --- TEST predictions for this fold
    test_pred_xgb_folds.append(xgb.predict(te_ohe))
    test_pred_cat_folds.append(cat.predict(te_ohe))

    print(f"Fold {fold}: XGB best_iter={getattr(xgb,'best_iteration',None)}, Cat best_iter={cat.get_best_iteration()}")

# -----------------------------
# 8) Meta model (Ridge on OOF)
# -----------------------------
oof_meta = np.vstack([oof_pred_xgb, oof_pred_cat]).T
ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge.fit(oof_meta, y_train_full)

# Average base TEST preds across folds, then meta-predict
test_pred_xgb = np.mean(test_pred_xgb_folds, axis=0)
test_pred_cat = np.mean(test_pred_cat_folds, axis=0)
test_meta = np.vstack([test_pred_xgb, test_pred_cat]).T
test_pred_final_log = ridge.predict(test_meta)

# -----------------------------
# 9) Evaluate (invert log)
# -----------------------------
def eval_scores(y_true_log, y_pred_log, name="Model"):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} on TEST ===")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2:   {r2:.6f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# Base models’ TEST (for reference)
print("\nReference (per-fold avg) base models on TEST:")
_ = eval_scores(y_test, test_pred_xgb, name="XGBoost (CV-avg)")
_ = eval_scores(y_test, test_pred_cat, name="CatBoost (CV-avg)")

# Stacked final
final_scores = eval_scores(y_test, test_pred_final_log, name="Stacked (XGB+Cat -> Ridge)")

# -----------------------------
# 10) Optional: save meta features & importance viz
# -----------------------------
try:
    # simple importance proxy: corr of OOF preds vs true
    corr_xgb = np.corrcoef(oof_pred_xgb, y_train_full)[0,1]
    corr_cat = np.corrcoef(oof_pred_cat, y_train_full)[0,1]
    print(f"\nOOF correlation — XGB: {corr_xgb:.4f}, Cat: {corr_cat:.4f}")
except Exception as e:
    pass

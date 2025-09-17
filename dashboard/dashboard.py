"""
Streamlit Dashboard for Tech Salary Models

Features:
- Load latest saved model from models/ (timestamped joblib files)
- Replicate preprocessing (manual OHE) to ensure proper feature alignment
- Evaluate on consistent 75/25 train/test split and display MAE/RMSE/R2
- Plot Train/Test Actual vs Predicted, and top-20 feature importances or coefficients
- Single-input form and CSV batch prediction

Run:
    streamlit run dashboard_app.py
"""

import streamlit as st
import toml
import os

# Connect config for theme
config_path = "../.streamlit/config.toml"

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        custom_config = toml.load(f)
    st.write("Custom setting:", custom_config.get("my_section", {}).get("my_setting"))

from glob import glob
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from category_encoders import TargetEncoder

CSV_PATH = "data/tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

NUM_FEATS = ["yearsofexperience", "yearsatcompany"]
CAT_FEATS = ["company", "title", "country", "gender", "Race", "Education"]


# ---------- Utilities ----------
def list_models_for(base_name: str, models_dir: str = "models") -> List[str]:
    pattern = os.path.join(models_dir, f"{base_name}_*.joblib")
    files = sorted(glob(pattern))
    # Exclude sidecars like *_te.joblib or *_cols.joblib
    files = [f for f in files if not (f.endswith("_te.joblib") or f.endswith("_cols.joblib"))]
    return files


def latest_model_for(base_name: str, models_dir: str = "models") -> Optional[str]:
    files = list_models_for(base_name, models_dir)
    return files[-1] if files else None


def load_dataset(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=[TARGET]).copy()
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    num = [c for c in NUM_FEATS if c in df.columns]
    cat = [c for c in CAT_FEATS if c in df.columns]
    X = df[num + cat].copy()
    # Keep target on original scale; we handle log transforms at prediction time
    y = df[TARGET].astype(float)
    return X, y


def infer_target_encoded_cols(feat_names: List[str], high_card_cols: List[str]) -> List[str]:
    """Infer which columns were target-encoded in training.

    If a raw column name (e.g., 'company') exists in feature names and there are
    no one-hot columns beginning with 'company_', assume target encoding.
    """
    te_cols = []
    for c in high_card_cols:
        has_plain = c in feat_names
        has_ohe = any(fn.startswith(f"{c}_") for fn in feat_names)
        if has_plain and not has_ohe:
            te_cols.append(c)
    return te_cols


def get_model_feature_names(model, fallback_columns: Optional[List[str]] = None) -> Optional[List[str]]:
    # Try scikit-learn convention
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass
    # CatBoost models often expose feature names via `feature_names_`
    if hasattr(model, "feature_names_"):
        try:
            names = list(getattr(model, "feature_names_"))
            if names:
                return names
        except Exception:
            pass
    # Another CatBoost approach: prettified feature importance includes names
    try:
        fi = getattr(model, "get_feature_importance", None)
        if callable(fi):
            df_imp = fi(prettified=True)
            if hasattr(df_imp, "__iter__"):
                # CatBoost may return list of dicts or DataFrame-like
                if hasattr(df_imp, "to_dict"):
                    cols = df_imp.columns.tolist()
                    name_col = "Feature Id" if "Feature Id" in cols else ("Feature" if "Feature" in cols else None)
                    if name_col:
                        names = list(df_imp[name_col])
                        if names:
                            return names
                else:
                    # List of dicts
                    try:
                        names = [d.get("Feature Id") or d.get("Feature") for d in df_imp]
                        names = [n for n in names if n is not None]
                        if names:
                            return names
                    except Exception:
                        pass
    except Exception:
        pass
    # Try XGBoost booster feature names
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    return list(fallback_columns) if fallback_columns is not None else None


def align_columns(X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return X.reindex(columns=columns, fill_value=0)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    # Compatibility with older scikit-learn: compute RMSE via sqrt(MSE)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ---------- UI ----------
st.set_page_config(page_title="Tech Salary Predictive Model Dashboard", layout="wide")
st.markdown(
"""
<style>
.app-title {
font-size: 26px;
line-height: 1.15;
margin: 0.25rem 0 0.25rem 0;
font-weight: 700;
text-align: center;
}
</style>
""",
unsafe_allow_html=True,
)
st.markdown("---")
st.markdown('<div class="app-title">T E C H⠀ S A L A R Y⠀ P R E D I C T I V E⠀ M O D E L⠀ D A S H B O A R D</div>', unsafe_allow_html=True)
st.markdown("---")

# Navigation sidebar as plain clickable text links using query params
from urllib.parse import quote
sections = [
    "A B O U T",
    "M O D E L",
    "F E A T U R E⠀ I M P O R T A N C E",
    "S I N G L E⠀ P R E D I C T I O N",
    "B A T C H⠀ P R E D I C T I O N",
]
try:
    params = st.query_params
    section = params.get("section", "A B O U T")
    if isinstance(section, list):
        section = section[0] if section else "A B O U T"
except Exception:
    params = st.experimental_get_query_params()
    section = params.get("section", ["A B O U T"])[0]
if section not in sections:
    section = "A B O U T"

# Render plain-text looking links that navigate in-place (no new tab)
nav_items = []
for label in sections:
    if label == section:
        nav_items.append(f'<div class="active">{label}</div>')
    else:
        nav_items.append(f'<a href="?section={quote(label)}" target="_self">{label}</a>')
nav_html = (
    """
<style>
/* Center nav vertically on the sidebar */
.sidebar-nav { min-height: 75vh; display: flex; flex-direction: column; justify-content: center; gap: 10px; }
@media (max-height: 800px) { .sidebar-nav { min-height: 70vh; } }

.sidebar-nav a { color: inherit !important; text-decoration: none !important; display:block; padding:10px 0; }
.sidebar-nav a:hover { text-decoration: none !important; color: #1DB954 !important; }
.sidebar-nav .active { font-weight: 700; padding:10px 0; color: #1DB954; }
</style>
<div class=\"sidebar-nav\">"""
    + "\n".join(nav_items)
    + "</div>"
)
st.sidebar.markdown(nav_html, unsafe_allow_html=True)

import plotly.express as px

base_name = "catboost"
model_path = latest_model_for(base_name)

if not model_path or not os.path.exists(model_path):
    st.warning("No saved CatBoost model found in models/. Train and save one first.")
    st.stop()

model = load(model_path)

# Try to load sidecar artifacts saved next to the model (for TE + column order)
sidecar_base, _ = os.path.splitext(model_path)
te_sidecar_path = sidecar_base + "_te.joblib"
cols_sidecar_path = sidecar_base + "_cols.json"
te_loaded = None
cols_loaded = None
try:
    if os.path.exists(te_sidecar_path):
        te_loaded = load(te_sidecar_path)
except Exception:
    te_loaded = None
try:
    if os.path.exists(cols_sidecar_path):
        with open(cols_sidecar_path, "r", encoding="utf-8") as f:
            cols_loaded = json.load(f)
except Exception:
    cols_loaded = None

# Detect if the loaded model already returns predictions on original scale
# (e.g., TransformedTargetRegressor with inverse_func applied)
is_ttr = isinstance(model, TransformedTargetRegressor) or (
    hasattr(model, "regressor_") and hasattr(model, "inverse_func")
)
# Heuristic for CatBoost/XGBoost models trained on log1p
model_outputs_log = bool(
    not is_ttr and ("xgcatboost" in os.path.basename(model_path).lower() or "catboost" in os.path.basename(model_path).lower())
)

# Load dataset and build train/test for evaluation
df = load_dataset(CSV_PATH)
X_all, y_all = preprocess(df)

# Discover feature names the model expects
feat_names = get_model_feature_names(
    model,
    fallback_columns=list(X_all.columns),
)
if cols_loaded:
    # Override with the exact training column order captured alongside the model
    feat_names = cols_loaded
if feat_names is None:
    st.error("Could not determine model feature names for alignment.")
    st.stop()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42
)

# Determine if training used target encoding for high-card columns
high_card_cols = [c for c in ["company", "title"] if c in X_all.columns]
te_cols = infer_target_encoded_cols(feat_names, high_card_cols) if 'infer_target_encoded_cols' in globals() else []
ohe_cols = [c for c in CAT_FEATS if c in X_all.columns and c not in te_cols]

# Fit target encoder on training (use log1p target if model outputs log)
te_encoder = None
te_from_sidecar = False
if te_cols:
    if te_loaded is not None:
        # Use the encoder saved with the model for exact mapping
        te_encoder = te_loaded
        X_train_te = te_encoder.transform(X_train_raw[te_cols])
        X_test_te = te_encoder.transform(X_test_raw[te_cols])
        te_from_sidecar = True
    else:
        te_encoder = TargetEncoder(cols=te_cols)
        # model_outputs_log may be defined later; default False here and updated below
        y_for_te_base = np.log1p(y_train) if False else y_train
        X_train_te = te_encoder.fit_transform(X_train_raw[te_cols], y_for_te_base)
        X_test_te = te_encoder.transform(X_test_raw[te_cols])
else:
    X_train_te = pd.DataFrame(index=X_train_raw.index)
    X_test_te = pd.DataFrame(index=X_test_raw.index)

# One-hot encode remaining categoricals (train + test to capture all levels; align later)
if ohe_cols:
    dummies = pd.get_dummies(
        pd.concat([X_train_raw[ohe_cols], X_test_raw[ohe_cols]], axis=0), drop_first=True
    )
    X_train_ohe = dummies.iloc[: len(X_train_raw)].reset_index(drop=True)
    X_test_ohe = dummies.iloc[len(X_train_raw) :].reset_index(drop=True)
    X_train_ohe.index = X_train_raw.index
    X_test_ohe.index = X_test_raw.index
else:
    X_train_ohe = pd.DataFrame(index=X_train_raw.index)
    X_test_ohe = pd.DataFrame(index=X_test_raw.index)

# Numeric features
num_cols = [c for c in NUM_FEATS if c in X_all.columns]
X_train_num = X_train_raw[num_cols]
X_test_num = X_test_raw[num_cols]

# Combine
X_train = pd.concat([X_train_num, X_train_te, X_train_ohe], axis=1)
X_test = pd.concat([X_test_num, X_test_te, X_test_ohe], axis=1)

# Align to model feature order
X_train_aligned = align_columns(X_train, feat_names)
X_test_aligned = align_columns(X_test, feat_names)

# No post-processing UI; predictions are shown as-is on the original scale

# If target encoding is needed, and the toggle indicates log outputs,
# refit the target encoder to log1p(y) to match training behavior.
if 'te_encoder' in globals() and te_encoder is not None and 'te_cols' in globals() and te_cols and not te_from_sidecar:
    y_for_te = np.log1p(y_train) if model_outputs_log else y_train
    try:
        te_encoder = te_encoder.fit(X_train_raw[te_cols], y_for_te)
        X_train[te_cols] = te_encoder.transform(X_train_raw[te_cols])
        X_test[te_cols] = te_encoder.transform(X_test_raw[te_cols])
        X_train_aligned = align_columns(X_train, feat_names)
        X_test_aligned = align_columns(X_test, feat_names)
    except Exception:
        pass

# Evaluation
train_preds = model.predict(X_train_aligned)
test_preds = model.predict(X_test_aligned)

# If model outputs log1p, invert to original USD for metrics/display
if model_outputs_log:
    train_preds = np.expm1(train_preds)
    test_preds = np.expm1(test_preds)
train_mae, train_rmse, train_r2 = compute_metrics(y_train, train_preds)
test_mae, test_rmse, test_r2 = compute_metrics(y_test, test_preds)

if section == "A B O U T":
    st.markdown(
        "<div style='text-align:justify;'>"
        "This dashboard is part of a project done by Garrison Goh Zen Ken of Asia Pacific "
        "University as a Final Year Project. It utilizes a CatBoost machine learning model, "
        "which is a type of gradient boosting algorithm. The model has been trained on a cleaned " 
        "dataset of salaries of individuals who work in the technology sector, to predict yearly " 
        "compensation based on various features, mainly to find the difference in pay between Male "
        "and Female individuals. The dashboard allows users to explore the model's performance "
        "and feature importance, make single predictions, as well as batch predictions."
        "</div>",
        unsafe_allow_html=True
    )

if section == "M O D E L":
    # Bar charts (Train/Test) with R² cards directly underneath each
    st.subheader("Model Performance Metrics")
    col_train, col_test = st.columns([1, 1])

    with col_train:
        df_train_metrics = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [train_mae, train_rmse]})
        fig_train = px.bar(df_train_metrics, x="Metric", y="Value", title="Train Metrics", text="Value",
                           color_discrete_sequence=["#1DB954"]
        )
        fig_train.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        ymax = df_train_metrics["Value"].max() * 1.15
        fig_train.update_yaxes(range=[0, ymax]) 
        fig_train.update_layout(yaxis_title="", xaxis_title="", height=350)
        st.plotly_chart(fig_train, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center; font-size:15px; font-weight:bold;'>Train R²</div>"
            f"<div style='text-align:center; font-size:20px;'>{train_r2:.4f}</div>",
            unsafe_allow_html=True
        )

    with col_test:
        df_test_metrics = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [test_mae, test_rmse]})
        fig_test = px.bar(
            df_test_metrics, x="Metric", y="Value", title="Test Metrics", text="Value",
            color_discrete_sequence=["#2045FF"]
        )
        fig_test.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        ymax = df_test_metrics["Value"].max() * 1.15
        fig_test.update_yaxes(range=[0, ymax])
        fig_test.update_layout(yaxis_title="", xaxis_title="", height=350)
        st.plotly_chart(fig_test, use_container_width=True)
        st.markdown(
            "<div style='text-align:center; font-size:15px; font-weight:bold;'>Test R²</div>"
            f"<div style='text-align:center; font-size:20px; '>{test_r2:.4f}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Actual vs Predicted")
    min_y = float(min(y_train.min(), y_test.min(), train_preds.min(), test_preds.min()))
    max_y = float(max(y_train.max(), y_test.max(), train_preds.max(), test_preds.max()))

    df_plot = pd.DataFrame(
        {
            "Actual": pd.concat([y_train.reset_index(drop=True), y_test.reset_index(drop=True)]),
            "Predicted": np.concatenate([train_preds, test_preds]),
            "Split": ["Train"] * len(y_train) + ["Test"] * len(y_test),
        }
    )
    fig = px.scatter(
        df_plot,
        x="Actual",
        y="Predicted",
        color="Split",
        opacity=0.6,
        height=450,
        color_discrete_map={"Train": "#1DB954", "Test": "#2045FF"},
    )
    fig.add_shape(type="line", x0=min_y, y0=min_y, x1=max_y, y1=max_y, line=dict(dash="dash", color="black"))
    fig.update_layout(
        xaxis_range=[min_y, max_y],
        yaxis_range=[min_y, max_y],
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

if section == "F E A T U R E⠀ I M P O R T A N C E":
    st.subheader("Feature Importance")
    try:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feat_names)
        elif hasattr(model, "coef_"):
            importances = pd.Series(np.abs(model.coef_), index=feat_names)
        else:
            importances = None
    except Exception:
        importances = None

    if importances is not None:
        topk = importances.sort_values(ascending=False).head(20)
        # Plot as horizontal bar chart in descending order
        fig_imp = px.bar(
            topk.sort_values(ascending=True),   # ascending True so biggest appears on top
            x=topk.sort_values(ascending=True),
            y=topk.sort_values(ascending=True).index,
            orientation="h",
            title="Top 20 Features",
            labels={"x": "Importance", "y": "Feature"},
            color_discrete_sequence=["#1DB954"],
        )
        fig_imp.update_layout(height=600, yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("This model does not expose importances/coefficients.")

    st.markdown(
        "<div style='text-align:justify;'> " 
        "As shown above, the most important features include years of experience, " 
        "years at the company, job title, and company name. The analysis of the dataset " 
        "used shows that the Gender feature does not significantly impact the compensation " 
        "prediction, which also means there is only a slight pay gap between Male and Female "
        "in the tech industry."
        "</div>"
        ,unsafe_allow_html=True
    )

    # ===============================
    # Gender vs Compensation (visual + correlation)
    # ===============================
    st.markdown("---")
    st.subheader("Gender vs Total Yearly Compensation")

    # Find gender column (case-insensitive)
    gcol = None
    for cand in ["gender", "Gender"]:
        if cand in df.columns:
            gcol = cand
            break

    if gcol is None:
        st.error("No 'gender' column found in the dataset.")
    else:
        # Prepare / clean
        df_g = df[[gcol, TARGET]].dropna().copy()
        normalize_map = {
            "M": "Male", "F": "Female",
            "male": "Male", "female": "Female",
            "MALE": "Male", "FEMALE": "Female",
        }
        df_g[gcol] = df_g[gcol].map(lambda x: normalize_map.get(str(x), str(x)))
        order = df_g[gcol].value_counts().index.tolist()

        # Violin + box (nice overview of spread)
        fig_v = px.violin(
            df_g, x=gcol, y=TARGET, points="outliers", box=True,
            category_orders={gcol: order},
            title="Distribution of Compensation by Gender",
            labels={gcol: "Gender", TARGET: "Total Yearly Compensation"},
            color=gcol,
            color_discrete_map={"Male": "#2045FF", "Female": "#FA7B7B"}
        )
        fig_v.update_layout(height=500)
        st.plotly_chart(fig_v, use_container_width=True)

        # Summary stats
        summary = (
            df_g.groupby(gcol)[TARGET]
            .agg(n="count", mean="mean", median="median", std="std")
            .sort_values("n", ascending=False)
        )   
        st.markdown("<div style='font-size:15px; font-weight:bold;'> Summary Statistics by Gender </div>", unsafe_allow_html=True)
        st.markdown("")
        st.dataframe(summary)
        st.markdown("")

        # Correlation: point-biserial if binary, else one-vs-rest table
        st.markdown("<div style='font-size:15px; font-weight:bold;'>"
                    "Correlation (point-biserial)"
                    "</div>"
                    "<div style='font-size:12px; color:gray;'>"
                    "between Gender (Male/Female) and totalyearlycompensation "
                    "</div>", 
                    unsafe_allow_html=True)
        st.markdown("")

        def _point_biserial(binary_series, y):
            try:
                from scipy.stats import pointbiserialr
                r, p = pointbiserialr(binary_series, y)
                return float(r), float(p)
            except Exception:
                # Fallback: Pearson on {0,1} vs y
                r = np.corrcoef(binary_series.astype(float), y.astype(float))[0, 1]
                return float(r), None

        cats = df_g[gcol].dropna().unique().tolist()
        if len(cats) == 2:
            # map larger group -> 0, smaller -> 1 for stability
            cats_sorted = sorted(cats, key=lambda c: (df_g[gcol] == c).sum(), reverse=True)
            bin_map = {cats_sorted[0]: 0, cats_sorted[1]: 1}
            b = df_g[gcol].map(bin_map)
            r_pb, pval = _point_biserial(b, df_g[TARGET])

            pbs, pvalue = st.columns([1, 1])

            with pbs:
                st.markdown(
                    "<div style='text-align:center; font-size:15px; font-weight:bold;'>"
                    f"Point-biserial r"
                    "</div>"
                    "<div style='text-align:center; font-size:20px;'>"
                    f"{r_pb:.4f}"
                    "</div>",
                    unsafe_allow_html=True
                )
                st.markdown("")
                st.markdown(
                    "<div style='text-align:justify;'>"
                    "The point biserial correlation coefficient (r) measures the strength and direction " 
                    "of the relationship between a continuous variable "
                    "and a dichotomous variable (a variable with only two categories) "
                    "It is a special case of the Pearson correlation, ranging from -1.00 to +1.00, " 
                    "where values close to -1 or +1 indicate a strong relationship, and a value of 0 " 
                    "indicates no relationship. In this case, the r value is close to 0, indicating" 
                    "a very weak relationship."
                    "</div>"
                    ,unsafe_allow_html=True
                )
            with pvalue:
                st.markdown(
                    "<div style='text-align:center; font-size:15px; font-weight:bold;'>"
                    "p-value"
                    "</div>"
                    "<div style='text-align:center; font-size:20px;'>"
                    f"{(f'{pval:.4g}') if pval is not None else ''}"
                    "</div>",
                    unsafe_allow_html=True
                )

                st.markdown("")
                st.markdown(
                    "<div style='text-align:justify;'>"
                    "A p-value is the probability of obtaining your observed results, or " 
                    "even more extreme results, if the null hypothesis were true. The null " 
                    "hypothesis usually states there is no difference or effect.  In this "
                    "case, Since the p-value is greater than 0.05, the results are considered " 
                    "statistically non-significant. "
                    "</div>",
                    unsafe_allow_html=True
                    )
                 
        else:
            rows = []
            for cat in order:
                b = (df_g[gcol] == cat).astype(int)
                r_pb, pval = _point_biserial(b, df_g[TARGET])
                rows.append({
                    "Category": cat,
                    "point_biserial_r (one-vs-rest)": r_pb,
                    "p_value": pval if pval is not None else "—",
                    "n_in_category": int((df_g[gcol] == cat).sum()),
                })
            out_df = pd.DataFrame(rows).sort_values("point_biserial_r (one-vs-rest)", ascending=False)
            st.dataframe(out_df, use_container_width=True)

# ---------- Single Prediction ----------
if section == "S I N G L E⠀ P R E D I C T I O N":
    st.subheader("Single Prediction")
    st.markdown("")
    st.markdown(
        "<div style='text-align:justify;'>"
        "This section allows you to input individual data points to predict the total yearly compensation of an individual. "
        "</div>"
        ,unsafe_allow_html=True
    )
    st.markdown("")


    with st.form("single_input_form"):
        c1, c2 = st.columns(2)
        with c1:
            yoexp = st.number_input("Years of experience", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
            yocomp = st.number_input("Years at company", min_value=0.0, max_value=60.0, value=1.0, step=0.5)
            title = st.selectbox("Title", sorted(df["title"].dropna().unique().tolist())) if "title" in df else ""
            company = st.selectbox("Company", sorted(df["company"].dropna().unique().tolist())) if "company" in df else ""
        with c2:
            # Robust lookup for country column (case-insensitive)
            _country_col = "country" if "country" in df.columns else ("Country" if "Country" in df.columns else None)
            country = (
                st.selectbox("Country", sorted(df[_country_col].dropna().unique().tolist()))
                if _country_col is not None
                else ""
            )
            gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique().tolist())) if "gender" in df else ""
            race = st.selectbox("Race", sorted(df["Race"].dropna().unique().tolist())) if "Race" in df else ""
            edu = st.selectbox("Education", sorted(df["Education"].dropna().unique().tolist())) if "Education" in df else ""

        submitted = st.form_submit_button("Predict Yearly Compensation")

if section == "S I N G L E⠀ P R E D I C T I O N" and submitted:
    row = {
        "yearsofexperience": yoexp,
        "yearsatcompany": yocomp,
        "company": company,
        "title": title,
        "gender": gender,
        "Race": race,
        "Education": edu,
        "country": country
    }
    X_row = pd.DataFrame([row])
    # Apply same preprocessing
    num = [c for c in NUM_FEATS if c in X_row.columns]
    cat = [c for c in CAT_FEATS if c in X_row.columns]
    # Target-encode TE columns if present
    X_parts = [X_row[num]]
    if 'te_cols' in globals() and te_cols:
        y_te_tmp = np.log1p(y_train) if model_outputs_log else y_train
        try:
            te_encoder = te_encoder.fit(X_train_raw[te_cols], y_te_tmp)
            X_parts.append(te_encoder.transform(X_row[[c for c in te_cols if c in X_row.columns]]))
            cat = [c for c in cat if c not in te_cols]
        except Exception:
            pass
    # One-hot the remaining categoricals; keep all levels then align
    X_row_ohe = pd.get_dummies(X_row[cat], drop_first=False)
    X_parts.append(X_row_ohe)
    X_row_enc = pd.concat(X_parts, axis=1)
    X_row_aligned = align_columns(X_row_enc, feat_names)
    pred = float(model.predict(X_row_aligned)[0])
    if model_outputs_log:
        pred = float(np.expm1(pred))
    st.success(f"Predicted total yearly compensation: {pred:,.2f}")


if section == "B A T C H⠀ P R E D I C T I O N":
    st.subheader("Batch Prediction (CSV)")
    st.markdown("")
    st.markdown(
        "<div style='text-align:justify;'>"
        "This section allows you to input a batch of data to predict the total yearly compensation of multiple individuals. "
        "</div>"
        ,unsafe_allow_html=True
    )
    st.write("Upload a CSV with columns: yearsofexperience, yearsatcompany, company, title, Country, gender, Race, Education")
    batch_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="batch")
else:
    batch_file = None
if section == "B A T C H⠀ P R E D I C T I O N" and batch_file is not None:
    try:
        # Show upload success
        st.info("File uploaded successfully. Processing...")
        
        # Load the CSV
        df_in = pd.read_csv(batch_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_in.head())
        
        # Show available columns
        st.write("Columns found in uploaded file:", list(df_in.columns))
        st.write("Expected columns:", NUM_FEATS + CAT_FEATS)
        
        # Keep only relevant columns if present
        cols = [c for c in NUM_FEATS + CAT_FEATS if c in df_in.columns]
        if not cols:
            st.error("No expected columns found in uploaded file.")
        else:
            st.success(f"Found {len(cols)} matching columns: {cols}")
            
            # For batch, replicate TE+OHE pipeline then align
            Xb = df_in[cols].copy()
            parts = [Xb[[c for c in NUM_FEATS if c in Xb.columns]]]
            
            # Get model output type from earlier in the code
            model_outputs_log = bool(
                not is_ttr and ("xgcatboost" in os.path.basename(model_path).lower() or "catboost" in os.path.basename(model_path).lower())
            )
            
            # Handle target encoding if needed
            if 'te_cols' in globals() and te_cols:
                y_te_tmp = np.log1p(y_train) if model_outputs_log else y_train
                try:
                    te_encoder = te_encoder.fit(X_train_raw[te_cols], y_te_tmp)
                    parts.append(te_encoder.transform(Xb[[c for c in te_cols if c in Xb.columns]]))
                except Exception as e:
                    st.error(f"Error during target encoding: {str(e)}")
                    st.stop()
            
            # Handle one-hot encoding for remaining categorical features
            oh = [c for c in CAT_FEATS if c in Xb.columns and (('te_cols' in globals() and c not in te_cols) or 'te_cols' not in globals())]
            if oh:
                parts.append(pd.get_dummies(Xb[oh], drop_first=False))
            
            # Combine all parts and align columns
            Xb = pd.concat(parts, axis=1)
            Xb_aligned = align_columns(Xb, feat_names)
            
            # Make predictions
            st.info("Making predictions...")
            preds = model.predict(Xb_aligned)
            
            # Convert predictions if needed
            if model_outputs_log:
                preds = np.expm1(preds)
            
            # Create output dataframe
            out = df_in.copy()
            out["predicted_totalyearlycompensation"] = preds
            
            # Show results preview
            st.success("Predictions completed!")
            st.write("Preview of predictions (first 20 rows):")
            st.dataframe(out.head(20))
            
            # Offer download button
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download complete predictions",
                data=csv_bytes,
                file_name=f"salary_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()
        oh = [c for c in CAT_FEATS if c in Xb.columns and (('te_cols' in globals() and c not in te_cols) or 'te_cols' not in globals())]
        if oh:
            parts.append(pd.get_dummies(Xb[oh], drop_first=False))
        Xb = pd.concat(parts, axis=1)
        Xb_aligned = align_columns(Xb, feat_names)
        preds = model.predict(Xb_aligned)
        if model_outputs_log:
            preds = np.expm1(preds)
        out = df_in.copy()
        out["predicted_totalyearlycompensation"] = preds
        st.dataframe(out.head(20))
        # Offer download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", data=csv_bytes, file_name=f"predictions_{base_name}.csv", mime="text/csv")

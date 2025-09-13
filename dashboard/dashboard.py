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

import os
from glob import glob
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH = "data/tech_salary_data_CLEANED.csv"
TARGET = "totalyearlycompensation"

NUM_FEATS = ["yearsofexperience", "yearsatcompany"]
CAT_FEATS = ["title", "location", "gender", "Race", "Education"]


# ---------- Utilities ----------
def list_models_for(base_name: str, models_dir: str = "models") -> List[str]:
    pattern = os.path.join(models_dir, f"{base_name}_*.joblib")
    return sorted(glob(pattern))


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
    y = df[TARGET].astype(float)
    X_enc = pd.get_dummies(X, columns=cat, drop_first=True)
    return X_enc, y


def get_model_feature_names(model, fallback_columns: Optional[List[str]] = None) -> Optional[List[str]]:
    # Try scikit-learn convention
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
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
st.set_page_config(page_title="Tech Salary Models Dashboard", layout="wide")
st.title("Tech Salary Models Dashboard")

# Appearance controls
st.sidebar.header("Appearance")
theme_choice = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
is_dark = theme_choice == "Dark"

theme_colors = (
    {
        "bg": "#0B1220",
        "panel": "#111827",
        "text": "#F9FAFB",
        "primary": "#60A5FA",
        "line": "#9CA3AF",
        "plotly_template": "plotly_dark",
    }
    if is_dark
    else {
        "bg": "#FFFFFF",
        "panel": "#F8FAFC",
        "text": "#111827",
        "primary": "#3B82F6",
        "line": "#374151",
        "plotly_template": "plotly_white",
    }
)

# Minimal CSS override to simulate theme switching at runtime
st.markdown(
    f"""
    <style>
      [data-testid="stAppViewContainer"] {{
        background-color: {theme_colors['bg']};
        color: {theme_colors['text']};
      }}
      [data-testid="stHeader"] {{
        background: {theme_colors['bg']};
      }}
      [data-testid="stSidebar"] > div {{
        background-color: {theme_colors['panel']};
      }}
      .stButton>button, .stDownloadButton>button {{
        background-color: {theme_colors['primary']};
        color: #ffffff;
      }}
      h1, h2, h3, h4, h5, h6, label, p, span {{
        color: {theme_colors['text']};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Model Selection")
model_presets = {
    "Baseline Linear Regression": "baseline_linear_regression",
    "Random Forest (No Tuning)": "random_forest_no_tuning",
    "Random Forest (RandomizedSearchCV)": "random_forest_randomizedsearchcv",
    "Random Forest (GridSearchCV)": "random_forest_gridsearchcv",
    "XGBoost (No Tuning)": "xgboost_no_tuning",
    "XGBoost (RandomizedSearchCV)": "xgboost_randomizedsearchcv",
    "XGBoost (GridSearchCV)": "xgboost_gridsearchcv",
}

preset = st.sidebar.selectbox("Choose model type", list(model_presets.keys()))
base_name = model_presets[preset]

latest_path = latest_model_for(base_name)
custom_model = st.sidebar.file_uploader("Or upload a .joblib model", type=["joblib"], accept_multiple_files=False)

model_path = None
if custom_model is not None:
    # Save to a temp file so joblib can load it
    tmp_path = os.path.join("models", f"uploaded_{custom_model.name}")
    os.makedirs("models", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(custom_model.getbuffer())
    model_path = tmp_path
else:
    model_path = latest_path

if not model_path or not os.path.exists(model_path):
    st.warning(f"No saved model found for '{preset}'. Train a model to models/ first.")
    st.stop()

st.sidebar.write(f"Loaded model: {os.path.basename(model_path)}")
model = load(model_path)

# Load dataset and build train/test for evaluation
df = load_dataset(CSV_PATH)
X_all, y_all = preprocess(df)

# Discover feature names the model expects, then align
feat_names = get_model_feature_names(model, fallback_columns=list(X_all.columns))
if feat_names is None:
    st.error("Could not determine model feature names for alignment.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42
)
X_train_aligned = align_columns(X_train, feat_names)
X_test_aligned = align_columns(X_test, feat_names)

# Optional post-processing to enforce non-negative salaries
st.sidebar.header("Post-processing")
enforce_nonneg = st.sidebar.checkbox("Enforce non-negative predictions", value=True)
lower_bound_mode = None
lower_bound_value = 0.0
if enforce_nonneg:
    lower_bound_mode = st.sidebar.selectbox(
        "Lower bound source",
        ["0", "Training minimum ($)"]
    )
    if lower_bound_mode == "Training minimum ($)":
        lower_bound_value = float(y_train.min())
    else:
        lower_bound_value = 0.0

# Evaluation
train_preds = model.predict(X_train_aligned)
test_preds = model.predict(X_test_aligned)
if enforce_nonneg:
    train_preds = np.maximum(train_preds, lower_bound_value)
    test_preds = np.maximum(test_preds, lower_bound_value)
train_mae, train_rmse, train_r2 = compute_metrics(y_train, train_preds)
test_mae, test_rmse, test_r2 = compute_metrics(y_test, test_preds)

left, right = st.columns(2)
with left:
    st.subheader("Model Performance Metrics")
    st.markdown(
        f"""
        - Train MAE: {train_mae:,.2f}
        - Train RMSE: {train_rmse:,.2f}
        - Train R2: {train_r2:.5f}
        - Test MAE: {test_mae:,.2f}
        - Test RMSE: {test_rmse:,.2f}
        - Test R2: {test_r2:.5f}
        """
    )

with right:
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
        height=400,
        color_discrete_map={
            "Train": "#3B82F6",  # Blue
            "Test": "#EC4899",   # Pink
        },
        template=theme_colors["plotly_template"],
    )
    fig.add_shape(
        type="line",
        x0=min_y,
        y0=min_y,
        x1=max_y,
        y1=max_y,
        line=dict(dash="dash", color=theme_colors["line"]),
    )
    fig.update_layout(xaxis_range=[min_y, max_y], yaxis_range=[min_y, max_y])
    st.plotly_chart(fig, use_container_width=True)

# Importance / coefficients
st.subheader("Top 20 Feature Importance / Coefficients")
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
    topk_sorted = topk.sort_values()
    fig_imp = px.bar(
        x=topk_sorted.values,
        y=topk_sorted.index,
        orientation="h",
        height=420,
        template=theme_colors["plotly_template"],
    )
    fig_imp.update_traces(marker_color="#3B82F6")  # Blue bars
    fig_imp.update_layout(xaxis_title="Importance / |Coefficient|", yaxis_title="Feature")
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("This model does not expose importances/coefficients.")


# ---------- Single Prediction ----------
st.header("Single Prediction")
with st.form("single_input_form"):
    c1, c2 = st.columns(2)
    with c1:
        yoexp = st.number_input("Years of experience", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
        yocomp = st.number_input("Years at company", min_value=0.0, max_value=60.0, value=1.0, step=0.5)
        title = st.selectbox("Title", sorted(df["title"].dropna().unique().tolist())) if "title" in df else ""
    with c2:
        location = st.selectbox("Location", sorted(df["location"].dropna().unique().tolist())) if "location" in df else ""
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique().tolist())) if "gender" in df else ""
        race = st.selectbox("Race", sorted(df["Race"].dropna().unique().tolist())) if "Race" in df else ""
        edu = st.selectbox("Education", sorted(df["Education"].dropna().unique().tolist())) if "Education" in df else ""

    submitted = st.form_submit_button("Predict Compensation")

if submitted:
    row = {
        "yearsofexperience": yoexp,
        "yearsatcompany": yocomp,
        "title": title,
        "location": location,
        "gender": gender,
        "Race": race,
        "Education": edu,
    }
    X_row = pd.DataFrame([row])
    # Apply same preprocessing
    num = [c for c in NUM_FEATS if c in X_row.columns]
    cat = [c for c in CAT_FEATS if c in X_row.columns]
    # Important: do NOT drop_first for single rows; otherwise the only
    # observed category is dropped and categoricals become all zeros.
    X_row_enc = pd.get_dummies(X_row[num + cat], columns=cat, drop_first=False)
    X_row_aligned = align_columns(X_row_enc, feat_names)
    pred = float(model.predict(X_row_aligned)[0])
    if enforce_nonneg:
        pred = float(max(pred, lower_bound_value))
    st.success(f"Predicted total yearly compensation: {pred:,.2f}")


# ---------- Batch Prediction ----------
st.header("Batch Prediction (CSV)")
st.write("Upload a CSV with columns: yearsofexperience, yearsatcompany, title, location, gender, Race, Education")
batch_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="batch")
if batch_file is not None:
    df_in = pd.read_csv(batch_file)
    # Keep only relevant columns if present
    cols = [c for c in NUM_FEATS + CAT_FEATS if c in df_in.columns]
    if not cols:
        st.error("No expected columns found in uploaded file.")
    else:
        # For batch, also avoid drop_first so categories are retained,
        # then align to the training columns (which used drop_first).
        Xb = pd.get_dummies(df_in[cols], columns=[c for c in CAT_FEATS if c in cols], drop_first=False)
        Xb_aligned = align_columns(Xb, feat_names)
        preds = model.predict(Xb_aligned)
        if enforce_nonneg:
            preds = np.maximum(preds, lower_bound_value)
        out = df_in.copy()
        out["predicted_totalyearlycompensation"] = preds
        st.dataframe(out.head(20))
        # Offer download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions", data=csv_bytes, file_name=f"predictions_{base_name}.csv", mime="text/csv")


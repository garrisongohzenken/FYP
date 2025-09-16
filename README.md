# Tech Salary Prediction Project

An end-to-end machine learning project focused on predicting tech industry salaries using various regression models. The project includes comprehensive data preprocessing, exploratory data analysis, multiple model implementations, and a Streamlit dashboard for interactive predictions.

## Project Structure
- `data/` - Raw and cleaned salary datasets
- `notebooks/`
  - `data_preparation/` - Data preprocessing and exploratory analysis notebooks
  - `model_building/` - Various model implementations and experiments
    - Baseline models (Linear Regression)
    - Tree-based models (Random Forest, XGBoost)
    - LightGBM implementations
    - CatBoost implementation
    - Model utilities and helpers
- `models/` - Saved model artifacts and configurations
- `dashboard/` - Streamlit dashboard for model deployment
- `requirements.txt` - Project dependencies

## Features
- Multiple model implementations:
  - Baseline Linear Regression
  - Random Forest (with GridSearch optimization)
  - XGBoost (with manual tuning and HalvingSearch)
  - LightGBM (baseline and optimized versions)
  - CatBoost implementation
- Data processing:
  - Comprehensive exploratory data analysis
  - Feature engineering and preprocessing
  - Target encoding for categorical variables
- Model evaluation:
  - Train/test split validation
  - Multiple metrics (MAE, MSE, R²)
- Interactive dashboard:
  - Real-time predictions
  - Model performance visualization
  - Feature importance analysis

## Dependencies
Key requirements:
- streamlit >= 1.25
- pandas >= 1.5
- scikit-learn >= 1.2
- xgboost >= 1.7
- lightgbm >= 4.0
- catboost (latest version)
- category-encoders >= 2.6

See `requirements.txt` for complete list of dependencies.
  - `data/tech_salary_data_CLEANED.csv`
- `notebooks/` — EDA, preprocessing, and model building
  - `notebooks/data_preparation/exploratory_data_analysis.ipynb`
  - `notebooks/data_preparation/data_preprocessing.ipynb`
  - `notebooks/model_building/` (baseline, RF, XGB + tuning variants)
- `dashboard/` — Streamlit app
  - `dashboard/dashboard.py`
- `.streamlit/config.toml` — Theme (dark mode)
- `models/` — Saved models as timestamped `.joblib` files
- `requirements.txt` — Python dependencies

## Quickstart
1) Create and activate a virtual env
- Windows (PowerShell)
  - `python -m venv .venv`
  - `.venv\\Scripts\\Activate.ps1`
- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

2) Install dependencies
- `pip install -r requirements.txt`

3) Ensure data is available
- Place `tech_salary_data_CLEANED.csv` at `data/tech_salary_data_CLEANED.csv`.

4) Run the dashboard
- `streamlit run dashboard/dashboard.py`

## Using the Dashboard
- Model selection: choose a preset (Baseline, RF, XGBoost). The app loads the latest file in `models/` matching the preset name (e.g., `xgboost_no_tuning_YYYYMMDD_HHMMSS.joblib`).
- Custom model: optionally upload a `.joblib` file.
- Post‑processing: option to enforce non‑negative predictions (0 or training minimum).
- Metrics & plots: view Train/Test MAE, RMSE, R²; Actual vs Predicted scatter; top 20 importances/coefficients.
- Single prediction: fill out the form and get an instant estimate.
- Batch prediction (CSV): upload a CSV with columns:
  - `yearsofexperience, yearsatcompany, title, location, gender, Race, Education`
  - Download predictions as a CSV.

## Modeling Notes
- Preprocessing
  - Numeric: `yearsofexperience`, `yearsatcompany`
  - Categorical: `title`, `location`, `gender`, `Race`, `Education`
  - Training uses one‑hot encoding with `drop_first=True` and aligns columns to the model’s expected feature set.
  - Single/batch inference uses `drop_first=False` then reindexes to the training feature columns to avoid all‑zero categories for single rows.
- Evaluation
  - Fixed split: `train_test_split(..., test_size=0.25, random_state=42)`
  - Metrics: MAE, RMSE (sqrt MSE), R²
- Importance/coefficients
  - Tree/boosting models show `feature_importances_` (unitless, typically summing to 1).
  - Linear models show `|coef_|` (in target units per feature unit). This means scales aren’t directly comparable across model families.

## Training & Saving Models
- Use the notebooks under `notebooks/model_building/` to train models.
- Save artifacts to `models/` with the following name patterns so the dashboard can auto‑discover the latest:
  - `baseline_linear_regression_*.joblib`
  - `random_forest_no_tuning_*.joblib`
  - `random_forest_randomizedsearchcv_*.joblib`
  - `random_forest_gridsearchcv_*.joblib`
  - `xgboost_no_tuning_*.joblib`
  - `xgboost_randomizedsearchcv_*.joblib`
  - `xgboost_gridsearchcv_*.joblib`

## Theming
- The app uses a dark theme configured at `.streamlit/config.toml`. Adjust colors by editing hex values (primary, background, secondary background, text). Plotly figures use a matching dark template with blue/pink accents.

## Troubleshooting
- "No saved model found …": Train a model and save a `.joblib` under `models/` with a matching base name pattern.
- "Could not determine model feature names …": Ensure the saved model exposes feature names (e.g., `feature_names_in_` in scikit‑learn or booster feature names in XGBoost). The app will fall back to training columns when available.
- Mismatched columns at inference: The app reindexes features; ensure categorical values seen at inference were also seen during training for best results.

## Tech Stack
- Python, pandas, numpy
- scikit‑learn, XGBoost, joblib
- Streamlit, Plotly

## License
Add your preferred license here (e.g., MIT). If unspecified, this project remains unlicensed in public repositories.

## Contributing
Issues and PRs are welcome. For major changes, please open an issue first to discuss what you’d like to change.


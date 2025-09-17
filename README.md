# Tech Salary Prediction Project

A comprehensive end-to-end machine learning project focused on predicting tech industry salaries using various regression models. The project demonstrates a complete ML pipeline from data preprocessing through model deployment, featuring multiple optimized models and an interactive dashboard.

## Project Overview

This project implements and compares several regression models for tech salary prediction:
- Baseline Linear Regression (establishing fundamental performance metrics)
- Random Forest (with extensive cross-validation optimization)
- XGBoost (optimized using HalvingGridSearchCV for efficient parameter tuning)
- CatBoost (leveraging advanced gradient boosting techniques)

All models undergo thorough hyperparameter optimization using various cross-validation approaches (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV) to ensure optimal performance while maintaining computational efficiency.

## Project Structure
```
.
├── data/                      # Dataset directory
│   ├── tech_salary_data.csv          # Raw dataset
│   └── tech_salary_data_CLEANED.csv  # Preprocessed dataset
│
├── notebooks/
│   ├── data_preparation/             # Data processing notebooks
│   │   ├── data_preprocessing.ipynb
│   │   └── exploratory_data_analysis.ipynb
│   │
│   └── model_building/              # Model implementation
│       ├── baseline_model.py        # Linear regression baseline
│       ├── rf_backup.py            # Random Forest implementation
│       ├── xgcatboost.py           # CatBoost implementation
│       └── utils/                  # Helper functions
│
├── models/                    # Saved model artifacts
├── dashboard/                 # Streamlit interface
└── requirements.txt          # Project dependencies
```

## Features

### Data Processing
- Comprehensive exploratory data analysis
- Advanced feature engineering
- Handling of high-cardinality categorical variables
- Target encoding for complex categorical features
- Log transformation for better salary distribution modeling

### Model Implementation
- Multiple model architectures with different strengths:
  - Linear Regression (baseline performance)
  - Random Forest (robust non-linear modeling)
  - CatBoost (efficient handling of categorical variables)
- Extensive hyperparameter tuning
- Feature importance analysis
- Cross-validated performance metrics

### Model Evaluation
- Train/Test performance analysis
- Multiple evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - R² Score
- Actual vs Predicted visualizations
- Feature importance rankings

### Interactive Dashboard
- Real-time salary predictions
- Model performance comparisons
- Interactive feature importance visualization
- Support for single predictions and batch processing

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit dashboard:
```bash
cd dashboard
streamlit run dashboard.py
```

## Dependencies
Core requirements:
- Python >= 3.8
- pandas >= 1.5
- numpy >= 1.20
- scikit-learn >= 1.2
- catboost >= 1.2
- streamlit >= 1.25
- category-encoders >= 2.6
- matplotlib >= 3.5
- seaborn >= 0.12

See `requirements.txt` for complete list of dependencies.

## Model Performance

Each model has been evaluated on both training and test sets. The evaluation metrics include MAE, RMSE, and R² scores to provide a comprehensive view of model performance. Detailed performance metrics and comparisons can be found in the respective model implementation files.
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



# Multi-Variable Environmental Time-Series Forecasting and HVAC Optimization

**Course:** MMA3001 Numerical Methods and Machine Learning  
**Project Focus:** Predictive HVAC Thermal Management using Multivariate Environmental Sensor Telemetry  

---

## 🏗️ Project Overview
Modern commercial buildings rely heavily on HVAC systems, which account for a substantial share of total building energy consumption. Traditional reactive, thermostat-driven control leads to thermal overshoot and energy waste. This project implements a robust, reproducible machine learning pipeline to ingest raw multi-sensor telemetry, engineer temporal and lagging features, and generate multi-step temperature forecasts across heterogeneous thermal environments.

---

## 📂 Repository Structure

```text
mma3001-project/
├── data/
│   ├── processed/         # Cleaned parquet datasets (e.g., env_long.parquet)
│   └── raw/               # Raw sensor CSV logs and location metadata Excel files
├── docs/                  # Generated HTML code documentation (pdoc)
├── notebooks/
│   └── 01_audit_co2.ipynb # Exploratory data audits and CO2 analysis
├── outputs/               # Saved model figures, evaluation plots, and metrics
├── src/
│   ├── __init__.py
│   ├── data_loading.py    # Data ingestion and parquet serialization
│   ├── preprocessing.py   # Grid resampling and missing value handling
│   ├── features.py        # Lag features, rolling windows, and dataset builders
│   └── forecasting.py     # ML models, cross-validation, and evaluation metrics
├── tests/
│   ├── __init__.py
│   ├── test_features.py   # Unit tests for feature engineering
│   ├── test_forecasting.py# Unit tests for forecasting logic
│   └── test_preprocessing.py # Unit tests for data preprocessing
├── .gitignore             # Excludes large raw CSVs and cache files
├── LICENSE                # Open-source license (MIT)
├── README.md              # Project technical documentation
├── requirements.txt       # Python package dependencies
├── plot_feature_importance.py # Feature importance visualization script
├── plot_predictions.py    # Actual vs predicted trajectory plots
├── plot_residuals.py      # Residual analysis and error distribution plots
├── plot_robustness_curve.py # Model robustness and stability curves
├── run_cv.py              # Time-series cross-validation script
├── run_robustness.py      # Robustness evaluation script
└── run_test.py            # Final holdout test evaluation script
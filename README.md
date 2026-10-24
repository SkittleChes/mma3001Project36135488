# Multi-Step Temperature Forecasting Pipeline for HVAC Demand Control

**Course:** MMA3001 Numerical Methods and Machine Learning  
**Project Focus:** Predictive HVAC Thermal Management using Multivariate Environmental Sensor Telemetry  

---

## 🏗️ Project Overview
Modern commercial buildings rely heavily on HVAC systems, which account for up to 50% of total building energy consumption. Traditional reactive, thermostat-driven control leads to thermal overshoot and energy waste. This project implements a robust, reproducible machine learning pipeline to generate multi-step temperature forecasts ($3\text{h}$ and $6\text{h}$ horizons) across heterogeneous thermal environments, enabling proactive building automation and peak load shifting.

---

## 📂 Repository Structure

```text
mma3001-project/
├── data/
│   ├── processed/         # Cleaned parquet data (e.g., env_long.parquet)
│   └── raw/               # Raw sensor CSVs and location metadata
├── docs/                  # Project report and documentation
├── src/
│   ├── __init__.py
│   ├── data_loading.py    # Data ingestion and parqueting
│   ├── preprocessing.py   # Grid resampling and missing value handling
│   ├── features.py        # Lag features, rolling windows, and dataset builder
│   └── forecasting.py     # Model training, cross-validation, and holdout evaluation
├── tests/
│   ├── __init__.py
│   ├── test_features.py   # Unit tests for feature engineering
│   ├── test_forecasting.py# Unit tests for forecasting logic
│   └── test_preprocessing.py # Unit tests for data preprocessing
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run_cv.py              # Cross-validation execution script
└── run_test.py            # Final holdout evaluation script
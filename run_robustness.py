"""Sensor Dropout and Missing Data Stress-Testing Script

Evaluates model robustness by artificially injecting missing values (NaNs) into 
auxiliary environmental predictors across multiple dropout rates (0% to 50%).
"""

import numpy as np
from pathlib import Path
import pandas as pd

from src.features import make_dataset
from src.forecasting import fit_predict, make_models, score, split_train_test

# 1. Load data and configure test split for Sensor 869
data_path = Path("data/processed/env_long.parquet")
if not data_path.exists():
    raise FileNotFoundError(f"Processed data not found at {data_path}. Run data_loading.py first.")

df_long = pd.read_parquet(data_path)

sensor_id = 6012002000869
test_start_str = "2024-05-20"
horizon_steps = 18

print(f"Running Sensor Dropout Stress-Test for Sensor {sensor_id} (3h horizon)...")
sensor_df = df_long[df_long["sensorid"] == sensor_id].reset_index()
wide = sensor_df.pivot_table(index="time", columns="variable", values="value").sort_index()

X, y = make_dataset(wide, target="Temperature", horizon=horizon_steps, same_time_yesterday=True)

test_start_ts = pd.Timestamp(test_start_str)
if X.index.tz is not None and test_start_ts.tz is None:
    test_start_ts = test_start_ts.tz_localize(X.index.tz)

X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start_ts, horizon_steps)

# 2. Define dropout rates to simulate network packet loss / sensor failure
dropout_rates = [0.0, 0.1, 0.25, 0.50]
aux_predictors = ["CO2", "Humidity", "Pressure"]

robustness_results = []

for rate in dropout_rates:
    # Create a corrupted copy of test features
    X_te_stressed = X_te.copy()
    
    if rate > 0.0:
        np.random.seed(42)
        for col in aux_predictors:
            if col in X_te_stressed.columns:
                mask = np.random.rand(len(X_te_stressed)) < rate
                X_te_stressed.loc[mask, col] = np.nan

    # Train fresh model on clean training data, evaluate on stressed test data
    models = make_models()
    model = models["hist_gb"]
    pred, _, _ = fit_predict(model, X_tr, y_tr, X_te_stressed, target="Temperature")
    
    metrics = score(y_te, pred)
    robustness_results.append({
        "Dropout Rate (%)": int(rate * 100),
        "MAE (°C)": round(metrics["MAE"], 4),
        "RMSE (°C)": round(metrics["RMSE"], 4),
    })

robust_df = pd.DataFrame(robustness_results)
print("\n--- Stress-Test Results: Model Performance Under Sensor Dropout ---")
print(robust_df.to_string(index=False))
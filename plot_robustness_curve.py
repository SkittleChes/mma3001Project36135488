"""Sensor Dropout Robustness Curve Plotting Script

Visualizes model performance degradation (MAE & RMSE) across varying 
sensor dropout rates (0% to 50% missing data in auxiliary telemetry).
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.features import make_dataset
from src.forecasting import fit_predict, make_models, score, split_train_test

# 1. Load data and setup
data_path = Path("data/processed/env_long.parquet")
df_long = pd.read_parquet(data_path)

sensor_id = 6012002000869
test_start_str = "2024-05-20"
horizon_steps = 18

sensor_df = df_long[df_long["sensorid"] == sensor_id].reset_index()
wide = sensor_df.pivot_table(index="time", columns="variable", values="value").sort_index()

X, y = make_dataset(wide, target="Temperature", horizon=horizon_steps, same_time_yesterday=True)
test_start_ts = pd.Timestamp(test_start_str)
if X.index.tz is not None and test_start_ts.tz is None:
    test_start_ts = test_start_ts.tz_localize(X.index.tz)

X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start_ts, horizon_steps)

# 2. Run stress test across dropout rates
dropout_rates = [0.0, 0.05, 0.10, 0.25, 0.40, 0.50]
aux_predictors = ["CO2", "Humidity", "Pressure"]
maes, rmses = [], []

models = make_models()
model = models["hist_gb"]

for rate in dropout_rates:
    X_te_stressed = X_te.copy()
    if rate > 0.0:
        np.random.seed(42)
        for col in aux_predictors:
            if col in X_te_stressed.columns:
                mask = np.random.rand(len(X_te_stressed)) < rate
                X_te_stressed.loc[mask, col] = np.nan
                
    pred, _, _ = fit_predict(model, X_tr, y_tr, X_te_stressed, target="Temperature")
    metrics = score(y_te, pred)
    maes.append(metrics["MAE"])
    rmses.append(metrics["RMSE"])

# 3. Plot robustness curve with constrained layout and slightly smaller fonts
rates_percent = [int(r * 100) for r in dropout_rates]

plt.figure(figsize=(9, 5), dpi=300, constrained_layout=True)
plt.plot(rates_percent, maes, marker="o", linestyle="-", color="#1f77b4", linewidth=2, label="MAE (°C)")
plt.plot(rates_percent, rmses, marker="s", linestyle="--", color="#ff7f0e", linewidth=2, label="RMSE (°C)")

plt.title("Sensor 869: Robustness Under Sensor Dropout", fontsize=11, fontweight="bold", pad=12)
plt.xlabel("Auxiliary Feature Dropout Rate (%)", fontsize=10, labelpad=8)
plt.ylabel("Error Metric (°C)", fontsize=10, labelpad=8)
plt.xticks(rates_percent, fontsize=9)
plt.yticks(fontsize=9)
plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
plt.grid(True, linestyle=":", alpha=0.6)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "sensor_869_robustness_curve.png"
plt.savefig(plot_path, bbox_inches="tight")
print(f"Successfully generated and saved robustness curve to: {plot_path}")
plt.show()
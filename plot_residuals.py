"""Residual Error Distribution Analysis Script

Generates clean diagnostic plots examining model residuals (Actual minus Predicted) 
to evaluate error bias and variance over the test horizon.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import make_dataset
from src.forecasting import fit_predict, make_models, split_train_test

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

# 2. Train model and generate predictions
models = make_models()
model = models["hist_gb"]
pred_te, _, _ = fit_predict(model, X_tr, y_tr, X_te, target="Temperature")

# 3. Calculate residuals
residuals = y_te - pred_te

# 4. Create dual-panel residual plot with proper top margin
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Panel A: Residuals over time
axes[0].plot(X_te.index, residuals, color="#d62728", alpha=0.6, linewidth=1)
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Forecast Residuals Over Time ($Actual - Predicted$)", fontsize=11, fontweight="bold", pad=10)
axes[0].set_xlabel("Datetime (Australia/Melbourne)", fontsize=10, labelpad=8)
axes[0].set_ylabel("Residual Error (°C)", fontsize=10, labelpad=8)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].tick_params(axis='x', rotation=20)

# Panel B: Error distribution histogram
axes[1].hist(residuals, bins=50, color="#1f77b4", edgecolor="white", alpha=0.8, density=True)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Distribution of Prediction Errors", fontsize=11, fontweight="bold", pad=10)
axes[1].set_xlabel("Error Magnitude (°C)", fontsize=10, labelpad=8)
axes[1].set_ylabel("Density", fontsize=10, labelpad=8)
axes[1].grid(True, linestyle=":", alpha=0.6)

# Perfectly positioned main title and top margin
fig.suptitle("Sensor 869: Residual Diagnostics & Error Distribution (3h Horizon)", fontsize=13, fontweight="bold", y=0.96)
plt.subplots_adjust(top=0.85)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "sensor_869_residuals.png"
plt.savefig(plot_path, bbox_inches="tight")
print(f"Successfully generated and saved perfectly framed residual plot to: {plot_path}")
plt.show()
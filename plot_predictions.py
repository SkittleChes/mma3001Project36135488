"""Master Probabilistic & Point Forecasting Plotting Script (plot_predictions.py)

Generates a publication-quality, high-visibility dual-panel figure combining 
point forecasts against actual temperatures (Panel A) and probabilistic quantile 
confidence bands (Panel B) for Sensor 869. Optimized for readability across all screen sizes.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import make_dataset
from src.forecasting import make_models, make_quantile_models, split_train_test

# --- GLOBAL STYLING FOR SMALL-SCREEN READABILITY ---
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
})

# 1. Load data and setup Sensor 869
data_path = Path("data/processed/env_long.parquet")
if not data_path.exists():
    raise FileNotFoundError(f"Processed data not found at {data_path}. Run data_loading.py first.")

df_long = pd.read_parquet(data_path)

sensor_id = 6012002000869
test_start_str = "2024-05-20"
horizon_steps = 18  # 3-hour horizon (18 x 10-min steps)

print(f"Generating high-visibility probabilistic forecast plots for Sensor {sensor_id}...")
sensor_df = df_long[df_long["sensorid"] == sensor_id].reset_index()
wide = (
    sensor_df.pivot_table(index="time", columns="variable", values="value")
    .sort_index()
)

X, y = make_dataset(
    wide, target="Temperature", horizon=horizon_steps, same_time_yesterday=True
)

test_start_ts = pd.Timestamp(test_start_str)
if X.index.tz is not None and test_start_ts.tz is None:
  test_start_ts = test_start_ts.tz_localize(X.index.tz)

X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start_ts, horizon_steps)

# 2. Train models (Point forecast & Quantiles)
models = make_models()
y_tr_change = y_tr - X_tr["Temperature"]

# Point forecast model
model_point = models["hist_gb"]
model_point.fit(X_tr, y_tr_change)
pred_change_te = model_point.predict(X_te)
pred_te = X_te["Temperature"] + pred_change_te

# Quantile models for uncertainty bounds (Q10 and Q90)
q_models = make_quantile_models()
q_models["q10"].fit(X_tr, y_tr_change)
q_models["q90"].fit(X_tr, y_tr_change)

q10_te = X_te["Temperature"] + q_models["q10"].predict(X_te)
q90_te = X_te["Temperature"] + q_models["q90"].predict(X_te)

# 3. Zoom in on a clean 3-day evaluation window
zoom_days = 3
zoom_steps = zoom_days * 144  # 144 steps per day at 10-min intervals
eval_index = X_te.index[:zoom_steps]
actual_zoom = y_te.iloc[:zoom_steps]
pred_zoom = pred_te.iloc[:zoom_steps]
q10_zoom = q10_te.iloc[:zoom_steps]
q90_zoom = q90_te.iloc[:zoom_steps]

# 4. Create Dual-Panel Master Figure with automatic constrained layout spacing
fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=300, constrained_layout=True)

# --- PANEL A: Point Forecast vs Actuals ---
axes[0].plot(
    eval_index,
    actual_zoom,
    label="Actual Temperature",
    color="#2ca02c",
    linestyle="-",
    alpha=0.9,
)
axes[0].plot(
    eval_index,
    pred_zoom,
    label=r"Model Point Forecast ($\hat{y}$)",
    color="#1f77b4",
    linestyle="--",
    alpha=0.9,
)
axes[0].set_title(
    "A. Point Forecast Comparison (3-Hour Horizon)",
    fontweight="bold",
    pad=8,
    loc="left",
)
axes[0].set_ylabel(r"Temperature ($^\circ\text{C}$)", labelpad=8)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=0.95,
)
axes[0].tick_params(axis="x", rotation=10)

# --- PANEL B: Probabilistic Quantile Uncertainty Bands ---
axes[1].plot(
    eval_index,
    actual_zoom,
    label="Actual Temperature",
    color="#2ca02c",
    linestyle="-",
    alpha=0.9,
)
axes[1].plot(
    eval_index,
    pred_zoom,
    label=r"Median Forecast ($Q_{50}$)",
    color="#1f77b4",
    linestyle="--",
    alpha=0.9,
)
axes[1].fill_between(
    eval_index,
    q10_zoom,
    q90_zoom,
    color="#ff7f0e",
    alpha=0.25,
    label=r"80% Prediction Interval ($Q_{10} - Q_{90}$)",
)
axes[1].set_title(
    "B. Probabilistic Uncertainty Quantification & Quantile Bands",
    fontweight="bold",
    pad=8,
    loc="left",
)
axes[1].set_xlabel("Datetime (Australia/Melbourne)", labelpad=8)
axes[1].set_ylabel(r"Temperature ($^\circ\text{C}$)", labelpad=8)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=0.95,
)
axes[1].tick_params(axis="x", rotation=10)

# --- SUPER-TITLE ---
fig.suptitle(
    "Sensor 869: Multi-Step Thermal Forecasting & Risk Profiles",
    fontsize=16,
    fontweight="bold",
)

# Save high-resolution graphic
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "sensor_869_probabilistic_forecast.png"
plt.savefig(plot_path, bbox_inches="tight")
print(f"Successfully generated and saved publication-grade master plot to: {plot_path}")
plt.show()
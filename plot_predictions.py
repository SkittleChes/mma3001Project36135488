"""Combined Forecasting & Uncertainty Quantification Master Plot

Generated a professional dual-panel figure showing:
  - Top Panel: Point forecast comparison (Actual vs. HistGradientBoosting vs. Baseline)
  - Bottom Panel: Probabilistic uncertainty quantification (Q50 Median + Q10-Q90 Prediction Interval)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.features import make_dataset
from src.forecasting import make_models, make_quantile_models, split_train_test

# 1. Load data and setup Sensor 869 (3h horizon)
data_path = Path("data/processed/env_long.parquet")
if not data_path.exists():
    raise FileNotFoundError(f"Processed data not found at {data_path}. Run data_loading.py first.")

df_long = pd.read_parquet(data_path)
sensor_id = 6012002000869
test_start_str = "2024-05-20"
horizon_steps = 18

print(f"Generating combined point & quantile forecast visualization for Sensor {sensor_id}...")
sensor_df = df_long[df_long["sensorid"] == sensor_id].reset_index()
wide = sensor_df.pivot_table(index="time", columns="variable", values="value").sort_index()

X, y = make_dataset(wide, target="Temperature", horizon=horizon_steps, same_time_yesterday=True)

test_start_ts = pd.Timestamp(test_start_str)
if X.index.tz is not None and test_start_ts.tz is None:
    test_start_ts = test_start_ts.tz_localize(X.index.tz)

X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start_ts, horizon_steps)

# 2. Train standard point model (HistGradientBoosting)
models = make_models()
point_model = models["hist_gb"]
point_model.fit(X_tr, y_tr - X_tr["Temperature"])
point_pred = X_te["Temperature"] + point_model.predict(X_te)

# 3. Train quantile models (Q10, Q50, Q90)
q_models = make_quantile_models()
q_preds = {}
for q_name, model in q_models.items():
    model.fit(X_tr, y_tr - X_tr["Temperature"])
    q_preds[q_name] = X_te["Temperature"] + model.predict(X_te)

# 4. Assemble evaluation dataframe & select a clean 3-day visual window
eval_df = pd.DataFrame({
    "Actual": y_te,
    "PointPred": point_pred,
    "Persistence": X_te["Temperature"],  # Baseline comparison
    "Q10": q_preds["q10"],
    "Q50": q_preds["q50"],
    "Q90": q_preds["q90"],
}, index=X_te.index)

slice_steps = 72 * 6  # 3 days on a 10-minute grid
df_slice = eval_df.iloc[:slice_steps]

# 5. Create Dual-Panel Master Figure with perfect breathing room
fig, axes = plt.subplots(2, 1, figsize=(12, 9.5), dpi=300, sharex=True)

# --- Top Panel: Point Forecast Accuracy ---
axes[0].plot(df_slice.index, df_slice["Actual"], label="Actual Temperature", color="black", linewidth=1.5, alpha=0.8)
axes[0].plot(df_slice.index, df_slice["PointPred"], label="HistGradientBoosting (Point Forecast)", color="#1f77b4", linewidth=1.5)
axes[0].plot(df_slice.index, df_slice["Persistence"], label="Persistence Baseline", color="#ff7f0e", linestyle="--", linewidth=1, alpha=0.7)
axes[0].set_title("A. Deterministic Point Forecast vs. Operational Baseline (3h Horizon)", fontsize=11, fontweight="bold", pad=12)
axes[0].set_ylabel("Temperature (°C)", fontsize=10, labelpad=8)
axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper left")
axes[0].grid(True, linestyle=":", alpha=0.6)

# --- Bottom Panel: Probabilistic Uncertainty Bands ---
axes[1].plot(df_slice.index, df_slice["Actual"], label="Actual Temperature", color="black", linewidth=1.5, alpha=0.8)
axes[1].plot(df_slice.index, df_slice["Q50"], label="Median Forecast (Q50)", color="#2ca02c", linewidth=1.5)
axes[1].fill_between(
    df_slice.index, df_slice["Q10"], df_slice["Q90"], 
    color="#2ca02c", alpha=0.2, label="80% Prediction Interval [Q10 - Q90]"
)
axes[1].set_title("B. Probabilistic Uncertainty Bounds & Risk Quantiles", fontsize=11, fontweight="bold", pad=12)
axes[1].set_xlabel("Datetime (Australia/Melbourne)", fontsize=10, labelpad=8)
axes[1].set_ylabel("Temperature (°C)", fontsize=10, labelpad=8)
axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper left")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].tick_params(axis='x', rotation=15)

# Overall Title & Layout Adjustments (Pushed subplots down so title has ample room)
fig.suptitle("Sensor 869: Comprehensive Thermal Forecasting & Risk Quantification Suite", fontsize=12, fontweight="bold", y=0.96)
plt.subplots_adjust(top=0.82, hspace=0.35)

# Save to outputs directory
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "sensor_869_comprehensive_forecast.png"
plt.savefig(plot_path, bbox_inches="tight")
print(f"Successfully generated and saved master forecast plot to: {plot_path}")
plt.show()
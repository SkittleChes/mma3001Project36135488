"""Feature Importance Visualization Script

Computes and plots permutation feature importance for the HistGradientBoosting model 
to quantify which input variables drive multi-step temperature forecasts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features import make_dataset
from src.forecasting import fit_predict, make_models, split_train_test

# 1. Load data and setup Sensor 869
data_path = Path("data/processed/env_long.parquet")
df_long = pd.read_parquet(data_path)

sensor_id = 6012002000869
test_start_str = "2024-05-20"
horizon_steps = 18

print(f"Computing feature importance for Sensor {sensor_id} (3h horizon)...")
sensor_df = df_long[df_long["sensorid"] == sensor_id].reset_index()
wide = sensor_df.pivot_table(index="time", columns="variable", values="value").sort_index()

X, y = make_dataset(wide, target="Temperature", horizon=horizon_steps, same_time_yesterday=True)

test_start_ts = pd.Timestamp(test_start_str)
if X.index.tz is not None and test_start_ts.tz is None:
    test_start_ts = test_start_ts.tz_localize(X.index.tz)

X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start_ts, horizon_steps)

# 2. Train model using change formulation
models = make_models()
model = models["hist_gb"]
y_tr_change = y_tr - X_tr["Temperature"]
model.fit(X_tr, y_tr_change)

# 3. Calculate permutation importance on test set
result = permutation_importance(
    model, X_te, y_te - X_te["Temperature"], n_repeats=5, random_state=42, n_jobs=-1
)

# 4. Create sorted DataFrame for plotting
importance_df = pd.DataFrame({
    "Feature": X_te.columns,
    "Importance Mean": result.importances_mean,
    "Importance Std": result.importances_std,
}).sort_values(by="Importance Mean", ascending=True)

# Plot top 10 most important features
top_features = importance_df.tail(10)

plt.figure(figsize=(10, 6), dpi=300)
plt.barh(top_features["Feature"], top_features["Importance Mean"], xerr=top_features["Importance Std"], color="#1f77b4", alpha=0.85, capsize=4)
plt.title("Sensor 869: Top 10 Feature Importances (Permutation Analysis)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Mean Importance Score (Decrease in Model Accuracy)", fontsize=11, labelpad=8)
plt.ylabel("Feature Name", fontsize=11, labelpad=8)
plt.grid(True, axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "sensor_869_feature_importance.png"
plt.savefig(plot_path)
print(f"Successfully generated and saved feature importance plot to: {plot_path}")
plt.show()
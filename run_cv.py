"""Time-Series Cross-Validation and Model Benchmarking Script

This script loads processed long-format environmental telemetry data, builds 
wide-format resampled grids for Sensor 869 and Sensor 326, applies chronological 
holdout split rules to prevent data leakage, and evaluates forecasting models 
across multiple horizons using time-series cross-validation.
"""

from pathlib import Path
import pandas as pd

from src.features import make_dataset
from src.forecasting import cross_validate, make_models

# Define project data path for processed environment telemetry
DATA_PATH: Path = Path("data/processed/env_long.parquet")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed data not found at {DATA_PATH}. Run data_loading.py first."
    )

# Load the long-format environmental telemetry dataset
df_long: pd.DataFrame = pd.read_parquet(DATA_PATH)

# Iterate over target sensor IDs of interest for model evaluation
SENSOR_IDS: list[int] = [6012002000869, 6012002000326]

for sensor_id in SENSOR_IDS:
    print(f"\n==============================")
    print(f"Evaluating Sensor: {sensor_id}")
    print(f"==============================")

    # Filter telemetry for the current sensor and reset index for pivoting
    sensor_df: pd.DataFrame = df_long[df_long["sensorid"] == sensor_id].reset_index()

    # Pivot sensor data onto a uniform 10-minute wide grid indexed by time
    wide: pd.DataFrame = (
        sensor_df.pivot_table(
            index="time", columns="variable", values="value"
        )
        .sort_index()
    )

    # Apply chronological holdout split rules before cross-validation
    if sensor_id == 6012002000869:
        train_val_data: pd.DataFrame = wide.loc[: "2024-05-19"]
    else:
        train_val_data = wide.loc[: "2026-02-26"]

    # Define test horizons: 3h (18 steps) and 6h (36 steps) on a 10-minute grid
    horizons: list[tuple[str, int]] = [("3h", 18), ("6h", 36)]
    
    for horizon_name, horizon_steps in horizons:
        # Construct supervised feature matrix X and target vector y
        X, y = make_dataset(
            train_val_data,
            target="Temperature",
            horizon=horizon_steps,
            same_time_yesterday=True,
        )

        # Initialize fresh models and execute time-series cross-validation
        models = make_models()
        scores = cross_validate(X, y, models, n_splits=4, gap=horizon_steps)

        # Print performance summary metrics for the current horizon
        print(f"\n--- Horizon: {horizon_name} ---")
        for model_name, metrics in scores.items():
            print(
                f"  {model_name:<25}: MAE = {metrics['MAE_mean']:.3f} (±"
                f" {metrics['MAE_std']:.3f})"
            )
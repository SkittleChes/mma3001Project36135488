"""Final Holdout Evaluation and Benchmarking Script

This script loads processed long-format environmental telemetry data, builds 
wide-format resampled grids for Sensor 869 and Sensor 326, applies official 
chronological holdout test start dates, and executes the final holdout evaluation 
comparing machine learning models against operational baselines.
"""

from pathlib import Path
import pandas as pd

from src.features import make_dataset
from src.forecasting import final_test

# Define project data path for processed environment telemetry
DATA_PATH: Path = Path("data/processed/env_long.parquet")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed data not found at {DATA_PATH}. Run data_loading.py first."
    )

# Load the long-format environmental telemetry dataset
df_long: pd.DataFrame = pd.read_parquet(DATA_PATH)

# Define test configurations per sensor with official holdout start dates
sensor_configs: dict[int, dict[str, str]] = {
    6012002000869: {
        "test_start": "2024-05-20",
        "name": "Sensor 869 (Schedule-driven, Bldg 60)",
    },
    6012002000326: {
        "test_start": "2026-02-27",
        "name": "Sensor 326 (Weather-driven)",
    },
}

# Define test horizons: 3h (18 steps) and 6h (36 steps) on a 10-minute grid
horizons: list[tuple[str, int]] = [("3h", 18), ("6h", 36)]

# Iterate through each sensor and execute final holdout evaluation
for sensor_id, cfg in sensor_configs.items():
    print(f"\n==================================================")
    print(f"Final Holdout Evaluation: {cfg['name']}")
    print(f"==================================================")

    # Filter telemetry for the current sensor and reset index for pivoting
    sensor_df: pd.DataFrame = df_long[df_long["sensorid"] == sensor_id].reset_index()

    # Pivot sensor data onto a uniform 10-minute wide grid indexed by time
    wide: pd.DataFrame = (
        sensor_df.pivot_table(
            index="time", columns="variable", values="value"
        )
        .sort_index()
    )

    for horizon_name, horizon_steps in horizons:
        # Construct supervised feature matrix X and target vector y
        X, y = make_dataset(
            wide,
            target="Temperature",
            horizon=horizon_steps,
            same_time_yesterday=True,
        )

        # Align timezone of test_start timestamp with feature index if timezone-aware
        test_start_ts = pd.Timestamp(cfg["test_start"])
        if X.index.tz is not None and test_start_ts.tz is None:
            test_start_ts = test_start_ts.tz_localize(X.index.tz)

        # Execute final holdout test evaluation
        results_df = final_test(
            wide=wide,
            X=X,
            y=y,
            horizon=horizon_steps,
            test_start=test_start_ts,
            target="Temperature",
        )

        # Print final performance evaluation table for the current horizon
        print(f"\n--- Horizon: {horizon_name} (Test Start: {cfg['test_start']}) ---")
        print(results_df.to_string(index=False))
"""Feature Engineering and Supervised Dataset Construction Module

This module provides feature engineering utilities to transform wide-format 
environmental time-series data into supervised feature matrices (X) and target vectors (y) 
for multi-step forecasting across various prediction horizons.
"""

import numpy as np
import pandas as pd


def make_dataset(
    wide: pd.DataFrame,
    target: str = "Temperature",
    horizon: int = 6,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 36, 72, 144),
    predictors: tuple[str, ...] = ("CO2", "Humidity", "Pressure"),
    same_time_yesterday: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Constructs feature matrix X and target vector y for multi-horizon forecasting,

    incorporating lag features, auxiliary environmental predictors, cyclical time-of-day 
    encodings, and optional same-time-yesterday differential features.

    Parameters:
    -----------
    wide : pd.DataFrame
        Wide-format environmental DataFrame indexed by timestamps.
    target : str, default="Temperature"
        The target variable column name to forecast.
    horizon : int, default=6
        Number of steps ahead to forecast (e.g., 6 steps = 1 hour on a 10-minute grid).
    lags : tuple of int, default=(1, 2, 3, 6, 12, 36, 72, 144)
        Historical lag intervals (in grid steps) of the target variable to include as features.
    predictors : tuple of str, default=("CO2", "Humidity", "Pressure")
        Auxiliary environmental variables to include as concurrent features at time t.
    same_time_yesterday : bool, default=False
        Whether to include features capturing the value and change at the same time yesterday.

    Returns:
    --------
    tuple of (pd.DataFrame, pd.Series)
        Cleaned feature matrix X and target vector y with all NaN-touching rows removed.
    """
    df = wide.copy()

    # Define target variable as the future value h steps ahead
    y = df[target].shift(-horizon)

    features: dict[str, pd.Series] = {}

    # Include current baseline value of the target variable at time t
    features[target] = df[target]

    # Generate historical lag features for the target variable
    for lag in lags:
        features[f"{target}_lag_{lag}"] = df[target].shift(lag)

    # Include auxiliary environmental predictor variables at current time t if present
    for pred in predictors:
        if pred in df.columns:
            features[pred] = df[pred]

    # Extract robust datetime index for cyclical time-of-day feature engineering
    if isinstance(df.index, pd.DatetimeIndex):
        dt_idx = df.index
    elif "datetime" in df.columns:
        dt_idx = pd.to_datetime(df["datetime"])
    else:
        dt_idx = pd.to_datetime(df.index)

    # Compute cyclical sine and cosine transformations for hour of day
    hour = dt_idx.hour + dt_idx.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Incorporate same-time-yesterday features if enabled
    if same_time_yesterday:
        shift_amount = 144 - horizon
        if shift_amount > 0:
            val_yesterday_target = df[target].shift(shift_amount)
            val_now = df[target]
            features["temp_same_time_yesterday"] = val_yesterday_target
            features["temp_yesterday_diff"] = val_yesterday_target - val_now

    # Construct initial feature matrix dataframe
    X = pd.DataFrame(features, index=df.index)

    # Combine features and target, then drop all rows containing NaNs 
    # resulting from missing data gaps, lag offsets, or horizon shifts
    data = X.copy()
    data["y"] = y
    data = data.dropna()

    return data.drop(columns=["y"]), data["y"]
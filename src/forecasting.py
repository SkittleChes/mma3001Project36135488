"""Baselines, models and chronological evaluation for multi-step forecasting."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STEPS_PER_DAY = 144  # 10-minute grid
STEP = pd.Timedelta("10min")


def baseline_predictions(wide, X, target="Temperature", horizon=6):
    """Predictions of three baselines for each row of X.

    persistence: T(t).
    yesterday: T(t + horizon - 24 h), i.e. the value at the target time yesterday.
    persistence_plus_daily_change: T(t) plus the change yesterday saw over the
        same window, T(t + horizon - 24 h) - T(t - 24 h).

    Args:
        wide: Wide DataFrame on the 10-minute grid.
        X: Feature table from make_dataset (its index selects the rows).
        target: Column being forecast.
        horizon: Steps ahead.

    Returns:
        DataFrame with one column per baseline, indexed like X.
    """
    ts = wide[target]
    now = X[target]
    yday = ts.shift(STEPS_PER_DAY - horizon).reindex(X.index)
    yday_now = ts.shift(STEPS_PER_DAY).reindex(X.index)
    return pd.DataFrame({
        "persistence": now,
        "yesterday": yday,
        "persistence_plus_daily_change": now + (yday - yday_now),
    })


def score(y_true, y_pred):
    """Return MAE and RMSE (same units as the target) as a dict."""
    err = y_true - y_pred
    return {"MAE": float(err.abs().mean()), "RMSE": float(np.sqrt((err ** 2).mean()))}


def make_models():
    """Return fresh, untrained models keyed by name."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, random_state=0
        ),
    }


def fit_predict(model, X_tr, y_tr, X_te, target="Temperature"):
    """Fit on the change from the current value and return absolute predictions.

    Returns:
        (predictions, fit_seconds, predict_seconds).
    """
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr - X_tr[target])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = X_te[target] + model.predict(X_te)
    return pred, fit_s, time.perf_counter() - t0


def split_train_test(X, y, test_start, horizon):
    """Chronological split with an embargo.

    Training rows whose target time falls on or after test_start are removed,
    so no training label comes from the test period.
    """
    embargo = test_start - horizon * STEP
    train = X.index < embargo
    test = X.index >= test_start
    return X[train], y[train], X[test], y[test]


def cross_validate(X, y, models, n_splits=4, gap=0, target="Temperature"):
    """Evaluate models using TimeSeriesSplit with an optional gap.

    Returns:
        A dictionary mapping model names to metrics (mean/std MAE and RMSE).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    results = {name: {"MAE": [], "RMSE": []} for name in models}

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in models.items():
            # Fit and predict using change formulation
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr - X_tr[target])
            pred = X_val[target] + model.predict(X_val)
            
            metrics = score(y_val, pred)
            results[name]["MAE"].append(metrics["MAE"])
            results[name]["RMSE"].append(metrics["RMSE"])

    # Aggregate mean and std across folds
    summary = {}
    for name, metrics in results.items():
        summary[name] = {
            "MAE_mean": float(np.mean(metrics["MAE"])),
            "MAE_std": float(np.std(metrics["MAE"])),
            "RMSE_mean": float(np.mean(metrics["RMSE"])),
            "RMSE_std": float(np.std(metrics["RMSE"])),
        }
    return summary


def final_test(wide, X, y, horizon, test_start, target="Temperature"):
    """Train on everything before the test block and score on the test block once."""
    base = baseline_predictions(wide, X, target, horizon)
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start, horizon)
    rows = [{"model": n, **score(y_te, base.loc[X_te.index, n])} for n in base]
    for name, model in make_models().items():
        pred, fit_s, pred_s = fit_predict(model, X_tr, y_tr, X_te, target)
        rows.append({
            "model": name,
            **score(y_te, pred),
            "fit_s": fit_s,
            "predict_s": pred_s,
        })
    return pd.DataFrame(rows)

def make_quantile_models():
    """Return a dictionary of quantile regression models for uncertainty quantification."""
    return {
        "q10": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.1, max_iter=200, random_state=0
        ),
        "q50": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.5, max_iter=200, random_state=0
        ),
        "q90": HistGradientBoostingRegressor(
            loss="quantile", quantile=0.9, max_iter=200, random_state=0
        ),
    }
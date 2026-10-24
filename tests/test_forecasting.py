import numpy as np
import pandas as pd
from src.features import make_dataset
from src.forecasting import STEP, baseline_predictions, score, split_train_test


def _ramp(n=1000):
    idx = pd.date_range("2025-01-01", periods=n, freq="10min", tz="Australia/Melbourne")
    return pd.DataFrame({"Temperature": np.arange(n, dtype=float), "Carbon dioxide": 450.0,
                         "Humidity": 50.0, "Pressure": 1010.0}, index=idx)


def test_daily_change_baseline_exact_on_ramp():
    w = _ramp()
    X, y = make_dataset(w, horizon=6)
    base = baseline_predictions(w, X, horizon=6)
    assert np.allclose(base["persistence_plus_daily_change"], y)
    assert np.allclose(base["persistence"], y - 6)


def test_split_has_embargo():
    w = _ramp()
    X, y = make_dataset(w, horizon=6)
    test_start = w.index[800]
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, test_start, horizon=6)
    assert X_te.index.min() >= test_start
    assert X_tr.index.max() + 6 * STEP < test_start


def test_score_known_values():
    s = score(pd.Series([1.0, 3.0]), pd.Series([2.0, 1.0]))
    assert s["MAE"] == 1.5
    assert np.isclose(s["RMSE"], np.sqrt(2.5))
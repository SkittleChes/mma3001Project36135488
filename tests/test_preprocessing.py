import numpy as np
import pandas as pd
import pytest
from src.preprocessing import clean_series


def _df(times, values):
    return pd.DataFrame({"sensorid": 1, "variable": "Carbon dioxide",
                         "time": times, "value": values})


def test_duplicates_removed_and_grid_regular():
    t = pd.date_range("2025-01-01", periods=200, freq="10min", tz="UTC")
    df = pd.concat([_df(t, 500.0), _df(t[:50], 500.0)])
    out = clean_series(df, 1)
    assert len(out) == 200
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("10min")).all()


def test_short_gap_filled_long_gap_kept():
    t = pd.date_range("2025-01-01", periods=300, freq="10min", tz="UTC")
    vals = np.arange(300, dtype=float)
    keep = np.ones(300, bool)
    keep[100:102] = False          # 20 min gap -> should be filled
    keep[200:230] = False          # 5 h gap -> should stay NaN
    out = clean_series(_df(t[keep], vals[keep]), 1, min_block=10)
    assert out.iloc[100] == pytest.approx(100.0)
    assert out.iloc[215] != out.iloc[215]   # NaN

def test_tiny_isolated_block_dropped():
    t = pd.date_range("2025-01-01", periods=200, freq="10min", tz="UTC")
    glitch = pd.DatetimeIndex(["2024-06-01 00:00"], tz="UTC")   # lone stray reading
    df = pd.concat([_df(t, 500.0), _df(glitch, 999.0)])
    out = clean_series(df, 1)
    assert out.index.min() == t[0]
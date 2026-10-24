"""Turn raw sensor readings into a regular, gap-aware time series."""
import pandas as pd


def clean_series(df, sensor_id, variable="Carbon dioxide", freq="10min",
                 min_block=100, block_gap="60min", max_fill="30min"):
    """Build a regular series for one variable of one sensor.

    Args:
        df: Long-format env data with columns sensorid, variable, time, value.
        sensor_id: Sensor to extract.
        variable: Variable name, e.g. "Temperature" or "Carbon dioxide".
        freq: Grid spacing, e.g. "10min".
        min_block: Blocks of contiguous readings smaller than this are dropped
            (isolated points from clock glitches).
        block_gap: A gap longer than this starts a new block.
        max_fill: Gaps up to this length are linearly interpolated; longer
            gaps stay NaN.

    Returns:
        pandas.Series indexed by time on a regular grid, NaN where data is missing.
    """
    s = (df[(df.sensorid == sensor_id) & (df.variable == variable)]
         .drop_duplicates("time").sort_values("time").set_index("time")["value"])

    # Drop tiny isolated blocks
    block_id = (s.index.to_series().diff() > pd.Timedelta(block_gap)).cumsum()
    s = s[block_id.map(block_id.value_counts()) >= min_block]

    grid = s.resample(freq).mean()

    # Interpolate only NaN runs short enough to trust
    limit = int(pd.Timedelta(max_fill) / pd.Timedelta(freq))
    is_na = grid.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_len <= limit)
    return grid.where(~fillable, grid.interpolate(method="time"))

def build_wide(env, sensor_id,
               variables=("Temperature", "Carbon dioxide", "Humidity", "Pressure"),
               **clean_kwargs):
    """Build a table with one cleaned column per variable on a shared time grid.

    Args:
        env: Long-format env data (sensorid, variable, time, value).
        sensor_id: Sensor to extract.
        variables: Variable names to include; any the sensor never reported are skipped.
        **clean_kwargs: Passed to clean_series (freq, max_fill, ...).

    Returns:
        pandas.DataFrame indexed by time, one column per variable.
    """
    cols = {}
    for v in variables:
        if ((env.sensorid == sensor_id) & (env.variable == v)).any():
            cols[v] = clean_series(env, sensor_id, v, **clean_kwargs)
    return pd.DataFrame(cols)
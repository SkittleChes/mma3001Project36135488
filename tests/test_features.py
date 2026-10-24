import pandas as pd
from src.features import make_dataset


def test_make_dataset_with_same_time_yesterday():
  dates = pd.date_range("2024-01-01", periods=200, freq="10min")
  df = pd.DataFrame(
      {
          "Temperature": range(200),
          "Humidity": [50] * 200,
          "Pressure": [1013] * 200,
          "CO2": [400] * 200,
      },
      index=dates,
  )

  X, y = make_dataset(
      df, target="Temperature", horizon=6, same_time_yesterday=True
  )

  assert "temp_same_time_yesterday" in X.columns
  assert "temp_yesterday_diff" in X.columns
  assert len(X) == len(y)
  assert len(X) > 0
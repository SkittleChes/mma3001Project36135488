"""Environmental Telemetry Data Loading and Parsing Module

This module provides robust ingestion utilities for raw smart building environmental telemetry CSV files. 
It includes a parsing pipeline to flatten nested JSON environmental sensor readings into a clean, 
audited long-format pandas DataFrame stored as an optimized Parquet file.
"""

from collections.abc import Iterator
import json
from pathlib import Path
import pandas as pd

# Define target timezone for consistent temporal alignment
MELB: str = "Australia/Melbourne"


def flatten_env(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested JSON environmental telemetry data into a long-format DataFrame.

    Malformed JSON rows and individual readings missing 'time' or 'value' fields 
    are skipped and recorded in the returned DataFrame's .attrs dictionary for auditing.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe chunk containing 'sensorid' and nested 'jsondata' columns.

    Returns:
    --------
    pd.DataFrame
        A long-format DataFrame containing flattened sensor metrics with 
        audit counters stored in .attrs['bad_json_rows'] and .attrs['bad_readings'].
    """
    rows: list[tuple[int, str, str, int | float, float]] = []
    bad_json: int = 0
    bad_reading: int = 0
    
    # Iterate through sensor IDs and raw JSON strings
    for sid, js in zip(df["sensorid"], df["jsondata"]):
        try:
            payload = json.loads(js)
        except (TypeError, json.JSONDecodeError):
            bad_json += 1
            continue
            
        # Parse nested variable payloads and historical time-series values
        for var in payload:
            name = var.get("variable", {}).get("name")
            unit = var.get("variable", {}).get("unit")
            for v in var.get("values", []):
                if "time" not in v or "value" not in v:
                    bad_reading += 1
                    continue
                rows.append((sid, name, unit, v["time"], v["value"]))
                
    # Construct long-format DataFrame from flattened telemetry tuples
    out = pd.DataFrame(rows, columns=["sensorid", "variable", "unit", "time", "value"])
    
    # Convert epoch timestamps to datetime objects with Australia/Melbourne timezone
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True).dt.tz_convert(MELB)
    
    # Attach data quality audit metadata to DataFrame attributes
    out.attrs["bad_json_rows"] = bad_json
    out.attrs["bad_readings"] = bad_reading
    
    return out


def load_env(csv_path: str | Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Load and flatten raw environmental sensor CSV data across multiple chunks.

    Parameters:
    -----------
    csv_path : str or Path
        Path to the raw environmental telemetry CSV file.
    chunksize : int, default=50,000
        Number of rows to ingest per chunk to manage memory overhead during JSON parsing.

    Returns:
    --------
    pd.DataFrame
        A fully flattened long-format DataFrame containing all valid environmental sensor readings.
    """
    parts = [
        flatten_env(chunk) 
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype={"sensorid": "int64"})
    ]
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    # Define and ensure existence of the processed data output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate raw environmental data file, execute loading pipeline, and serialize directly to Parquet format
    raw_dir = Path("data/raw")
    env_csv = next(raw_dir.glob("5EnvSensor*.csv"))
    
    load_env(env_csv).to_parquet(output_dir / "env_long.parquet")
    print(f"Successfully processed and saved environmental data to: {output_dir / 'env_long.parquet'}")
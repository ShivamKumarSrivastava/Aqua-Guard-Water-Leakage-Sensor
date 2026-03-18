"""
modules/data_io.py
------------------
Save and load sensor DataFrames as CSV.
"""

import os
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_PATH = os.path.join(DATA_DIR, "sensor_data.csv")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str = DEFAULT_PATH) -> str:
    """Save DataFrame to CSV; returns the path written."""
    ensure_data_dir()
    df.to_csv(path, index=False)
    return path


def load_csv(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Load DataFrame from CSV. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved data found at: {path}")
    return pd.read_csv(path)


def data_exists(path: str = DEFAULT_PATH) -> bool:
    return os.path.exists(path)


def saved_file_info(path: str = DEFAULT_PATH) -> dict:
    """Return metadata about the saved file (size, modified time)."""
    if not os.path.exists(path):
        return {}
    stat = os.stat(path)
    return {
        "path":     path,
        "rows":     sum(1 for _ in open(path)) - 1,   # minus header
        "size_kb":  round(stat.st_size / 1024, 1),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }

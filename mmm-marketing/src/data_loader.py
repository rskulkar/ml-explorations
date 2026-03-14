"""Data loading and validation for the MMM dataset."""
from __future__ import annotations
import pathlib
from typing import Union
import pandas as pd

REQUIRED_COLUMNS = ["date_week", "y", "x1", "x2", "event_1", "event_2", "dayofyear", "t"]
KEY_COLUMNS = ["y", "x1", "x2"]

def load_raw(path: Union[str, pathlib.Path, object]) -> pd.DataFrame:
    """Load MMM CSV. Accepts file path or file-like object (e.g. io.StringIO)."""
    return pd.read_csv(path, parse_dates=["date_week"])

def validate(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return (is_valid, errors). Checks: required cols, nulls in key cols, monotonic dates."""
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    for col in [c for c in KEY_COLUMNS if c in df.columns]:
        n = df[col].isna().sum()
        if n:
            errors.append(f"Column '{col}' has {n} null/missing value(s)")
    if "date_week" in df.columns and not df["date_week"].is_monotonic_increasing:
        errors.append("date_week is not monotonically increasing")
    return (len(errors) == 0, errors)

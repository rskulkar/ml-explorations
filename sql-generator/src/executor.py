"""Run SQL against SQLite, return pandas DataFrame."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Union
import pandas as pd


def execute_query(sql: str, db_path: Union[str, Path]) -> pd.DataFrame:
    """Execute SQL and return results as DataFrame. Raises ValueError on bad SQL."""
    conn = sqlite3.connect(Path(db_path))
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as exc:
        raise ValueError(f"SQL execution failed.\nQuery: {sql}\nError: {exc}") from exc
    finally:
        conn.close()
    return df

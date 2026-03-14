"""Compare generated SQL against reference SQL."""
from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd
from executor import execute_query


def evaluate(generated_sql: str, reference_sql: str, db_path: Union[str, Path]) -> dict:
    """Return {match, generated_result_shape, reference_result_shape, results_match}."""
    match = generated_sql.strip().lower() == reference_sql.strip().lower()
    gen_df = execute_query(generated_sql, db_path)
    ref_df = execute_query(reference_sql, db_path)
    return {
        "match": match,
        "generated_result_shape": gen_df.shape,
        "reference_result_shape": ref_df.shape,
        "results_match": _dataframes_equal(gen_df, ref_df),
    }


def _dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    if df1.shape != df2.shape or set(df1.columns) != set(df2.columns):
        return False
    df1 = df1[sorted(df1.columns)]
    df2 = df2[sorted(df2.columns)]
    try:
        s1 = df1.sort_values(list(df1.columns)).reset_index(drop=True).astype(str)
        s2 = df2.sort_values(list(df2.columns)).reset_index(drop=True).astype(str)
        return s1.equals(s2)
    except Exception:
        return False

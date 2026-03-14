"""Extract SQLite schema for LLM prompts."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Union


def load_schema(db_path: Union[str, Path]) -> dict:
    """Return {table_name: {"columns": [...], "sample_rows": [...]}} for all tables."""
    conn = sqlite3.connect(Path(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [r[0] for r in cur.fetchall()]
        schema = {}
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = [{"name": r[1], "type": r[2]} for r in cur.fetchall()]
            cur.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_rows = [list(r) for r in cur.fetchall()]
            schema[table] = {"columns": columns, "sample_rows": sample_rows}
    finally:
        conn.close()
    return schema


def format_schema_context(schema_dict: dict) -> str:
    """Format schema dict as human-readable string for LLM context."""
    lines = []
    for table, meta in schema_dict.items():
        lines.append(f"TABLE {table}")
        col_str = ", ".join(f"{c['name']} ({c['type']})" for c in meta["columns"])
        lines.append(f"  Columns: {col_str}")
        if meta["sample_rows"]:
            lines.append("  Sample rows:")
            for row in meta["sample_rows"]:
                lines.append(f"    {row}")
        lines.append("")
    return "\n".join(lines)

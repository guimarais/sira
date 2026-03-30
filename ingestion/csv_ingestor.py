import re
import sqlite3
from pathlib import Path

import pandas as pd

from config import settings


def _sanitize_column(name: str) -> str:
    """Lowercase and replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def ingest_csv(filepath: str) -> dict:
    """Load a CSV file into the SQLite `stocks` table.

    Drops and recreates the table on each call so the schema always matches
    the uploaded file. Column names from the CSV are sanitised (lowercase,
    underscores). A `metadata` table stores the column list for use by the
    text-to-SQL retriever.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        dict with keys: rows_inserted, columns, status.

    Raises:
        ValueError: If the CSV is empty or cannot be parsed.
    """
    settings.ensure_dirs()
    path = Path(filepath)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Cannot read CSV '{path.name}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV '{path.name}' is empty.")

    df.columns = [_sanitize_column(c) for c in df.columns]
    columns = list(df.columns)

    con = sqlite3.connect(settings.sqlite_path)
    try:
        con.execute("DROP TABLE IF EXISTS stocks")
        con.execute("DROP TABLE IF EXISTS metadata")

        df.to_sql("stocks", con, index=False, if_exists="replace")

        con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        con.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("columns", ",".join(columns)),
        )
        con.commit()
    finally:
        con.close()

    return {"rows_inserted": len(df), "columns": columns, "status": "ok"}

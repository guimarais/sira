"""
Shared SQLite connection and schema management for SIRA.
"""
import sqlite3

from config import settings

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT    NOT NULL UNIQUE,
    type         TEXT    NOT NULL,
    ingested_at  TEXT    NOT NULL,
    chunks_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chroma_id    TEXT    NOT NULL,
    page         INTEGER,
    chunk_index  INTEGER NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection with foreign-key enforcement enabled."""
    con = sqlite3.connect(settings.sqlite_path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def ensure_schema() -> None:
    """Create documents and chunks tables if they do not already exist."""
    con = get_connection()
    try:
        for statement in _SCHEMA_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                con.execute(statement)
        con.commit()
    finally:
        con.close()

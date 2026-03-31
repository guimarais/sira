"""
Module for retrieving structured stock data via LLM-generated SQL queries.
"""
import re
import sqlite3
import anthropic

from config import settings

# Only SELECT statements are permitted — block any mutation or DDL.
_SAFE_SQL_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_DANGEROUS_RE = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _get_schema() -> tuple[str, str]:
    """Return (comma-separated column names, one sample row as string)."""
    con = sqlite3.connect(settings.sqlite_path)
    try:
        try:
            row = con.execute("SELECT value FROM metadata WHERE key='columns'").fetchone()
        except sqlite3.OperationalError:
            raise RuntimeError("No stocks table found. Upload a CSV first.")
        if row is None:
            raise RuntimeError("No stocks table found. Upload a CSV first.")
        columns = row[0]

        sample_row = con.execute("SELECT * FROM stocks LIMIT 1").fetchone()
        sample = str(dict(zip(columns.split(","), sample_row))) if sample_row else "{}"
    finally:
        con.close()
    return columns, sample


def _build_prompt(query: str, columns: str, sample: str, error: str | None = None) -> str:
    prompt = (
        """
        You are a SQLite expert. Output ONLY a single valid SQLite SELECT statement.
        No explanation, no markdown, no code fences.

        <SELECT_STATEMENT>
        Table name: stocks
        Columns: {columns}
        Sample row: {sample}
        Question: {query}
        </SELECT_STATEMENT>
        """
    )
    if error:
        prompt += f"\n\nThe previous query failed with: {error}\nFix it and output only the corrected SQL."
    return prompt


def _validate_sql(sql: str) -> None:
    """Raise ValueError if the SQL contains dangerous statements."""
    if not _SAFE_SQL_RE.match(sql):
        raise ValueError(f"Generated SQL does not start with SELECT: {sql!r}")
    if _DANGEROUS_RE.search(sql):
        raise ValueError(f"Generated SQL contains forbidden keywords: {sql!r}")


def _generate_sql(prompt: str) -> str:
    """Call Claude and return the raw SQL string."""
    response = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _execute_sql(sql: str) -> list[dict]:
    """Run the SQL and return rows as a list of dicts."""
    con = sqlite3.connect(settings.sqlite_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(sql).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def retrieve_structured(query: str) -> dict:
    """Generate and execute a SQL query against the stocks table.

    Uses Claude to translate the natural-language query into SQLite SQL.
    Validates the SQL against an allowlist, executes it, and retries once
    on failure with the error message appended to the prompt.

    Args:
        query: Natural-language question about the stock data.

    Returns:
        dict with keys:
            sql_generated (str): The SQL that was executed.
            rows (list[dict]): Result rows, empty list on failure.
            error (str | None): Error message if the query ultimately failed.
    """
    try:
        columns, sample = _get_schema()
    except RuntimeError as exc:
        return {"sql_generated": "", "rows": [], "error": str(exc)}

    prompt = _build_prompt(query, columns, sample)
    sql = ""

    # First attempt
    first_error: Exception | None = None
    try:
        sql = _generate_sql(prompt)
        _validate_sql(sql)
        rows = _execute_sql(sql)
        return {"sql_generated": sql, "rows": rows, "error": None}
    except (sqlite3.OperationalError, ValueError) as exc:
        first_error = exc

    # Single retry with error context
    retry_prompt = _build_prompt(query, columns, sample, error=str(first_error))
    try:
        sql = _generate_sql(retry_prompt)
        _validate_sql(sql)
        rows = _execute_sql(sql)
        return {"sql_generated": sql, "rows": rows, "error": None}
    except (sqlite3.OperationalError, ValueError) as second_error:
        return {"sql_generated": sql, "rows": [], "error": str(second_error)}

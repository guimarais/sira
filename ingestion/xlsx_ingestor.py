import tempfile
from pathlib import Path

import pandas as pd

from ingestion.csv_ingestor import ingest_csv


def ingest_xlsx(filepath: str, sheet: str | int = 0) -> dict:
    """Read one sheet from an XLSX file and ingest it as stock data.

    Converts the target sheet to a temporary CSV file and delegates to
    ingest_csv, so column sanitisation and registry writes are handled
    identically to a direct CSV upload.

    Args:
        filepath: Absolute or relative path to the XLSX file.
        sheet: Sheet name (str) or zero-based index (int). Defaults to 0.

    Returns:
        dict from ingest_csv plus a 'sheet' key with the resolved sheet name.

    Raises:
        ValueError: If the sheet does not exist or the data cannot be parsed.
    """
    path = Path(filepath)

    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Cannot read sheet {sheet!r} from '{path.name}': {exc}") from exc

    # Resolve the actual sheet name for reporting (when sheet was given as index)
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheet_names = xl.sheet_names
        resolved_sheet = sheet_names[sheet] if isinstance(sheet, int) else sheet
    except Exception:
        resolved_sheet = sheet

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    try:
        result = ingest_csv(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {**result, "sheet": resolved_sheet}

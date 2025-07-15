from pathlib import Path
from typing import List, Dict

import json
import pandas as pd

# --- existing PDF helper you copied over ---
from .pdf_loader import load_pdf


def _json_to_docs(path: Path) -> List[Dict]:
    """
    Expect a list/array of objects or NDJSON (one JSON obj per line)
    """
    docs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # NDJSON (newline‑delimited)
        if first_char and first_char not in "[{":
            for line in f:
                obj = json.loads(line)
                docs.append(
                    {
                        "text": json.dumps(obj, ensure_ascii=False, indent=2),
                        "meta": {"source_type": "json", "file_name": path.name},
                    }
                )
        else:  # normal JSON array or object
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    docs.append(
                        {
                            "text": json.dumps(obj, ensure_ascii=False, indent=2),
                            "meta": {"source_type": "json", "file_name": path.name},
                        }
                    )
            else:
                docs.append(
                    {
                        "text": json.dumps(data, ensure_ascii=False, indent=2),
                        "meta": {"source_type": "json", "file_name": path.name},
                    }
                )
    return docs


def _spreadsheet_to_docs(path: Path) -> List[Dict]:
    """
    Convert CSV / Excel cells to a single plain‑text blob per row.
    """
    df = (
        pd.read_csv(path)
        if path.suffix.lower() == ".csv"
        else pd.read_excel(path, sheet_name=None)  # all sheets
    )

    if isinstance(df, dict):  # excel returns dict of DataFrames
        frames = []
        for sheet_name, sheet_df in df.items():
            sheet_df["__sheet__"] = sheet_name
            frames.append(sheet_df)
        df = pd.concat(frames, ignore_index=True)

    docs: List[Dict] = []
    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(
            {
                "text": text,
                "meta": {
                    "source_type": "spreadsheet",
                    "file_name": path.name,
                },
            }
        )
    return docs


def ingest(path: Path) -> List[Dict]:
    """
    Return a list of dicts: {"text": str, "meta": {...}}
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in {".json", ".ndjson"}:
        return _json_to_docs(path)
    elif suffix in {".csv", ".xlsx", ".xls"}:
        return _spreadsheet_to_docs(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

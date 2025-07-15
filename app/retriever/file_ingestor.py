"""
app/retriever/file_ingestor.py
──────────────────────────────
Format‑agnostic loader used by unit‑tests and the RAG pipeline.

Returns: List[dict] – each element is
    {
        "text": "<chunk‑or‑row‑content>",
        "meta": {...rich metadata...}
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PyPDF2 import PdfReader

# ────────────────────────────────────────────────────────────────────────────────
# PDF
# ────────────────────────────────────────────────────────────────────────────────
def load_pdf(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    docs: List[Dict[str, Any]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            docs.append(
                {
                    "text": text,
                    "meta": {
                        "source_type": "pdf",
                        "file_name": path.name,
                        "page": page_num,
                    },
                }
            )
    return docs


# ────────────────────────────────────────────────────────────────────────────────
# CSV & Excel
# ────────────────────────────────────────────────────────────────────────────────
def load_csv_excel(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        frames = {"__csv__": pd.read_csv(path)}
        src_type = "spreadsheet" 
    else:  # .xlsx / .xls
        frames = pd.read_excel(path, sheet_name=None)
        src_type = "spreadsheet"

    docs: List[Dict[str, Any]] = []
    for sheet, df in frames.items():
        for idx, row in df.iterrows():
            text = " ".join(map(str, row.dropna().values))
            if text.strip():
                docs.append(
                    {
                        "text": text,
                        "meta": {
                            "source_type": src_type,
                            "file_name": path.name,
                            "sheet": None if sheet == "__csv__" else sheet,
                            "row": int(idx),
                        },
                    }
                )
    return docs


# ────────────────────────────────────────────────────────────────────────────────
# JSON & NDJSON
# ────────────────────────────────────────────────────────────────────────────────
def load_json(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".ndjson":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        records = json.loads(raw)
        if isinstance(records, dict):  # single object
            records = [records]

    docs: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        docs.append(
            {
                "text": json.dumps(rec, ensure_ascii=False),
                "meta": {
                    "source_type": "json",
                    "file_name": path.name,
                    "record": idx,
                },
            }
        )
    return docs


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────
def ingest(path: Path) -> List[Dict[str, Any]]:
    """
    One‑shot helper used by tests:

        docs = ingest(Path("sample.pdf"))
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in {".csv", ".xlsx", ".xls"}:
        return load_csv_excel(path)
    elif suffix in {".json", ".ndjson"}:
        return load_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

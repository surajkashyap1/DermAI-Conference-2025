from pathlib import Path
from app.retriever.file_ingestor import ingest

def test_pdf(tmp_path):
    # create a tiny PDF fixture or use one from repo
    docs = ingest(Path("tests/fixtures/sample.pdf"))
    assert docs and docs[0]["meta"]["source_type"] == "pdf"

def test_json(tmp_path):
    j = tmp_path / "sample.json"
    j.write_text('[{"a": 1, "b": 2}]', encoding="utf-8")
    docs = ingest(j)
    assert docs[0]["meta"]["source_type"] == "json"

def test_csv(tmp_path):
    c = tmp_path / "sample.csv"
    c.write_text("x,y\n3,4", encoding="utf-8")
    docs = ingest(c)
    assert docs[0]["meta"]["source_type"] == "spreadsheet"

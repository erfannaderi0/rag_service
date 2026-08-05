import json
from types import SimpleNamespace

import pytest

from pathlib import Path
import sys
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from app.ingestion import ingest as ingest_module


# ---------------------------------------------------------------------------
# load_and_chunk_pdf
# ---------------------------------------------------------------------------

def test_load_and_chunk_pdf_raises_when_pdf_has_no_extractable_text(monkeypatch):
    monkeypatch.setattr(ingest_module.PDFLoader, "load", lambda self, path: [])

    with pytest.raises(ValueError):
        ingest_module.load_and_chunk_pdf("fake.pdf")


def test_load_and_chunk_pdf_delegates_to_loader_and_chunker(monkeypatch):
    fake_pages = ["page-1", "page-2"]
    captured = {}

    monkeypatch.setattr(ingest_module.PDFLoader, "load", lambda self, path: fake_pages)

    def fake_chunk_document(self, pages, doc_id):
        captured["pages"] = pages
        captured["doc_id"] = doc_id
        return ["chunk-a", "chunk-b"]

    monkeypatch.setattr(ingest_module.TextChunker, "chunk_document", fake_chunk_document)

    result = ingest_module.load_and_chunk_pdf("fake.pdf")

    assert result == ["chunk-a", "chunk-b"]
    assert captured["pages"] == fake_pages
    assert isinstance(captured["doc_id"], str) and captured["doc_id"]  # a fresh uuid string


# ---------------------------------------------------------------------------
# store_chunks
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self):
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return _FakeCursor()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _make_chunk(content, chunk_id, metadata=None):
    return SimpleNamespace(content=content, id=chunk_id, metadata=metadata or {"page_number": 1})


def test_store_chunks_raises_on_length_mismatch():
    chunks = [_make_chunk("a", "id-1"), _make_chunk("b", "id-2")]
    with pytest.raises(RuntimeError):
        ingest_module.store_chunks(chunks, vectors=[[0.1, 0.2]])  # only 1 vector for 2 chunks


def test_store_chunks_builds_expected_rows_and_commits(monkeypatch):
    chunks = [_make_chunk("hello", "id-1", metadata={"page_number": 1})]
    vectors = [[0.1, 0.2, 0.3]]
    fake_conn = _FakeConnection()
    captured = {}

    def fake_execute_values(cur, query, rows, template):
        captured["rows"] = rows

    monkeypatch.setattr(ingest_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(ingest_module, "execute_values", fake_execute_values)

    ingest_module.store_chunks(chunks, vectors)

    (content, metadata_json, vector) = captured["rows"][0]
    assert content == "hello"
    assert json.loads(metadata_json) == {"page_number": 1, "chunk_id": "id-1"}
    assert vector == [0.1, 0.2, 0.3]
    assert fake_conn.committed is True
    assert fake_conn.closed is True


def test_store_chunks_rolls_back_and_closes_on_db_error(monkeypatch):
    chunks = [_make_chunk("hello", "id-1")]
    vectors = [[0.1]]
    fake_conn = _FakeConnection()

    def fake_execute_values(cur, query, rows, template):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(ingest_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(ingest_module, "execute_values", fake_execute_values)

    with pytest.raises(RuntimeError):
        ingest_module.store_chunks(chunks, vectors)

    assert fake_conn.rolled_back is True
    assert fake_conn.committed is False
    assert fake_conn.closed is True


# ---------------------------------------------------------------------------
# ingest_pdf
# ---------------------------------------------------------------------------

def test_ingest_pdf_embeds_chunk_contents_and_stores_them(monkeypatch):
    fake_chunks = [_make_chunk("first", "id-1"), _make_chunk("second", "id-2")]
    fake_vectors = [[1.0], [2.0]]
    stored = {}

    monkeypatch.setattr(ingest_module, "load_and_chunk_pdf", lambda pdf_path: fake_chunks)
    monkeypatch.setattr(
        ingest_module,
        "store_chunks",
        lambda chunks, vectors: stored.update(chunks=chunks, vectors=vectors),
    )

    embedder = SimpleNamespace(embed=lambda texts: fake_vectors if texts == ["first", "second"] else None)

    n_stored = ingest_module.ingest_pdf("fake.pdf", embedder)

    assert n_stored == 2
    assert stored["chunks"] == fake_chunks
    assert stored["vectors"] == fake_vectors


# ---------------------------------------------------------------------------
# is_already_ingested
# ---------------------------------------------------------------------------

class _FakeLookupCursor:
    def __init__(self, found):
        self._found = found
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params):
        self.executed_params = params

    def fetchone(self):
        return (1,) if self._found else None


class _FakeLookupConnection:
    def __init__(self, found):
        self.cursor_obj = _FakeLookupCursor(found)

    def cursor(self):
        return self.cursor_obj


def test_is_already_ingested_true_when_row_found():
    conn = _FakeLookupConnection(found=True)
    assert ingest_module.is_already_ingested("paper.pdf", conn) is True
    assert conn.cursor_obj.executed_params == ("paper.pdf",)


def test_is_already_ingested_false_when_no_row_found():
    conn = _FakeLookupConnection(found=False)
    assert ingest_module.is_already_ingested("paper.pdf", conn) is False


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

def test_ingest_directory_reports_when_no_pdfs_found(tmp_path, capsys):
    ingest_module.ingest_directory(tmp_path)
    assert "No PDFs found" in capsys.readouterr().out


def test_ingest_directory_skips_already_ingested_files(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.pdf").write_bytes(b"")

    fake_conn = _FakeConnection()
    monkeypatch.setattr(ingest_module, "Embedder", lambda **kwargs: object())
    monkeypatch.setattr(ingest_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(ingest_module, "is_already_ingested", lambda source_file, conn: True)

    def _should_not_be_called(pdf_path, embedder):
        raise AssertionError("ingest_pdf should not run for already-ingested files")

    monkeypatch.setattr(ingest_module, "ingest_pdf", _should_not_be_called)

    ingest_module.ingest_directory(tmp_path)

    out = capsys.readouterr().out
    assert "2 skipped" in out
    assert fake_conn.closed is True


def test_ingest_directory_ingests_new_files_and_closes_connection(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.pdf").write_bytes(b"")

    fake_conn = _FakeConnection()
    calls = []

    monkeypatch.setattr(ingest_module, "Embedder", lambda **kwargs: object())
    monkeypatch.setattr(ingest_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(ingest_module, "is_already_ingested", lambda source_file, conn: False)
    monkeypatch.setattr(ingest_module, "ingest_pdf", lambda pdf_path, embedder: calls.append(pdf_path) or 5)

    ingest_module.ingest_directory(tmp_path)

    out = capsys.readouterr().out
    assert len(calls) == 1
    assert "1 ingested" in out
    assert fake_conn.closed is True


def test_ingest_directory_continues_after_one_file_fails(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "b.pdf").write_bytes(b"")

    fake_conn = _FakeConnection()

    def flaky_ingest_pdf(pdf_path, embedder):
        if pdf_path.name == "a.pdf":
            raise ValueError("corrupt pdf")
        return 3

    monkeypatch.setattr(ingest_module, "Embedder", lambda **kwargs: object())
    monkeypatch.setattr(ingest_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(ingest_module, "is_already_ingested", lambda source_file, conn: False)
    monkeypatch.setattr(ingest_module, "ingest_pdf", flaky_ingest_pdf)

    ingest_module.ingest_directory(tmp_path)

    out = capsys.readouterr().out
    assert "1 ingested" in out
    assert "a.pdf" in out  # reported as failed
    assert fake_conn.closed is True


@pytest.mark.integration
def test_ingest_directory_against_real_database_and_model(tmp_path):
    """Needs a running Postgres and the real BGE-M3 model. Point this at a
    directory with at least one real PDF and run with:
        pytest --run-integration
    """
    ingest_module.ingest_directory("data/documents")

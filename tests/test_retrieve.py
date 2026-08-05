import pytest

from pathlib import Path
import sys
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    
from app import retrieve as retrieve_module


class _FakeCursor:
    """Minimal stand-in for a psycopg2 cursor used as a context manager."""

    def __init__(self, rows):
        self._rows = rows
        self.executed = None  # (query, params) from the last execute() call

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params):
        self.executed = (query, params)

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = _FakeCursor(rows)
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True


def test_retrieve_maps_db_rows_into_result_dicts(monkeypatch):
    fake_rows = [
        (1, "chunk one text", {"source_file": "a.pdf"}, 0.91),
        (2, "chunk two text", {"source_file": "b.pdf"}, 0.85),
    ]
    fake_conn = _FakeConnection(fake_rows)
    monkeypatch.setattr(retrieve_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retrieve_module._embedder, "embed", lambda texts: [[0.1, 0.2, 0.3]])

    results = retrieve_module.retrieve("what is X?", top_k=2)

    assert results == [
        {"id": 1, "content": "chunk one text", "metadata": {"source_file": "a.pdf"}, "similarity": 0.91},
        {"id": 2, "content": "chunk two text", "metadata": {"source_file": "b.pdf"}, "similarity": 0.85},
    ]


def test_retrieve_passes_embedded_query_vector_and_top_k_to_sql(monkeypatch):
    fake_conn = _FakeConnection(rows=[])
    monkeypatch.setattr(retrieve_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retrieve_module._embedder, "embed", lambda texts: [[9.0, 9.0]])

    retrieve_module.retrieve("query text", top_k=7)

    _query, params = fake_conn.cursor_obj.executed
    vector, vector_again, top_k = params
    assert vector == [9.0, 9.0]
    assert vector_again == [9.0, 9.0]  # embedding <=> %s appears twice: SELECT and ORDER BY
    assert top_k == 7


def test_retrieve_embeds_the_query_wrapped_in_a_list(monkeypatch):
    captured = {}

    def fake_embed(texts):
        captured["texts"] = texts
        return [[1.0]]

    fake_conn = _FakeConnection(rows=[])
    monkeypatch.setattr(retrieve_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retrieve_module._embedder, "embed", fake_embed)

    retrieve_module.retrieve("hello world")

    assert captured["texts"] == ["hello world"]


def test_retrieve_closes_connection_even_if_query_fails(monkeypatch):
    fake_conn = _FakeConnection(rows=[])

    def _boom(*args, **kwargs):
        raise RuntimeError("db exploded")

    fake_conn.cursor_obj.execute = _boom
    monkeypatch.setattr(retrieve_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retrieve_module._embedder, "embed", lambda texts: [[0.0]])

    with pytest.raises(RuntimeError):
        retrieve_module.retrieve("query")

    assert fake_conn.closed is True


def test_retrieve_returns_empty_list_when_no_matches(monkeypatch):
    fake_conn = _FakeConnection(rows=[])
    monkeypatch.setattr(retrieve_module, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retrieve_module._embedder, "embed", lambda texts: [[0.0]])

    assert retrieve_module.retrieve("nothing matches this") == []


@pytest.mark.integration
def test_retrieve_against_real_database_and_model():
    """Needs `docker compose up` (Postgres + pgvector) with at least one
    ingested document, and the real BGE-M3 model. Run with:
        pytest --run-integration
    """
    results = retrieve_module.retrieve("test query", top_k=1)
    assert isinstance(results, list)
    if results:
        assert set(results[0].keys()) == {"id", "content", "metadata", "similarity"}

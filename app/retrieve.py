import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from app import config
from app.db import get_connection
from app.ingestion.embedding import Embedder

# Loaded once at import time — BGEM3FlagModel is expensive to instantiate,
# don't create a new Embedder per call.
_embedder = Embedder()


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """
    Embed `query` with BGE-M3 and return the top_k most similar chunks
    from the documents table, ranked by cosine similarity (descending).
    """
    query_vector = _embedder.embed([query])[0]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"id": row[0], "content": row[1], "metadata": row[2], "similarity": row[3]}
        for row in rows
    ]


if __name__ == "__main__":

    query = " ".join(sys.argv[1:]) or input("Query: ")
    results = retrieve(query)

    for r in results:
        print(f"[{r['similarity']:.4f}] (id={r['id']}) {r['content'][:120]!r}")

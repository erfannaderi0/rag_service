from pathlib import Path
import uuid
import json
import os
import sys
from psycopg2.extras import execute_values

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from app.ingestion.loader import PDFLoader
from app.ingestion.chunking import TextChunker, Chunk
from app.ingestion.embedding import Embedder
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from app.db import get_connection


def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    loader = PDFLoader()
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    pages = loader.load(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")

    doc_id = str(uuid.uuid4())
    return chunker.chunk_document(pages, doc_id=doc_id)


def store_chunks(chunks: list[Chunk], vectors: list[list[float]]):
    if len(chunks) != len(vectors):
        raise RuntimeError("Embedding count mismatch with chunk count")

    rows = [
        (chunk.content, json.dumps({**chunk.metadata, "chunk_id": chunk.id}), vector)
        for chunk, vector in zip(chunks, vectors)
    ]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO documents (content, metadata, embedding) VALUES %s",
                rows,
                template="(%s, %s::jsonb, %s::vector)",
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ingest_pdf(pdf_path: str | Path, embedder: Embedder) -> int:
    """Ingest a single PDF. Returns number of chunks stored."""
    chunks = load_and_chunk_pdf(pdf_path)
    vectors = embedder.embed([c.content for c in chunks])
    store_chunks(chunks, vectors)
    return len(chunks)


def is_already_ingested(source_file: str, conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM documents WHERE metadata->>'source_file' = %s LIMIT 1",
            (source_file,)
        )
        return cur.fetchone() is not None


def ingest_directory(directory: str | Path, pattern: str = "*.pdf"):
    directory = Path(directory)
    pdf_files = sorted(directory.glob(pattern))

    if not pdf_files:
        print(f"No PDFs found in {directory} matching {pattern}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {directory}")
    embedder = Embedder(model_name=EMBEDDING_MODEL)  # load model once, reuse across all files

    conn = get_connection()
    succeeded, skipped, failed = 0, 0, []
    try:
        for pdf_path in pdf_files:
            if is_already_ingested(pdf_path.name, conn):
                print(f"  ⏭ {pdf_path.name}: already ingested, skipping")
                skipped += 1
                continue
            try:
                n_chunks = ingest_pdf(pdf_path, embedder)
                print(f"  ✓ {pdf_path.name}: {n_chunks} chunks")
                succeeded += 1
            except Exception as e:
                print(f"  ✗ {pdf_path.name}: {e}")
                failed.append(pdf_path.name)
    finally:
        conn.close()

    print(f"\nDone: {succeeded} ingested, {skipped} skipped, {len(pdf_files) - succeeded - skipped} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/documents"
    ingest_directory(target)

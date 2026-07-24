from pathlib import Path
import uuid
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from loader import PDFLoader
from chunking import TextChunker, Chunk
from embedding import Embedder
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = dict(
    dbname=os.getenv("POSTGRES_DB", "ragdb"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", 5432),
)

def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    loader = PDFLoader()
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    pages = loader.load(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")

    doc_id = str(uuid.uuid4())
    return chunker.chunk_document(pages, doc_id=doc_id)


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def store_chunks(chunks: list[Chunk], vectors: list[list[float]]):
    if len(chunks) != len(vectors):
        raise RuntimeError("Embedding count mismatch with chunk count")

    rows = [
        (chunk.content, json.dumps(chunk.metadata), vector)
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
    embedder = Embedder()  # load model once, reuse across all files

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
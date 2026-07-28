"""
generate_golden_qa.py

Generates a golden question/answer evaluation set (JSONL) for the RAG service
by sampling chunks from Postgres and asking an LLM (Groq llama-3.3-70b-versatile)
to produce a grounded, single-chunk factual Q&A pair for each sampled chunk.

Sampling strategy: N chunks per distinct source_file (even coverage across PDFs,
instead of a flat random sample which would bias toward longer documents).

Output format (one JSON object per line):
{
    "question": str,
    "ground_truth": str,          # the reference answer
    "contexts": [str],            # the single source chunk (RAGAS-compatible field name)
    "chunk_id": str,
    "source_file": str
}

Usage:
    python evaluation/generate_golden_qa.py --per-doc 4 --output evaluation/golden_qa.jsonl

Requires in .env:
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    GROQ_API_KEY
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from groq import Groq
from app.db import get_connection

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2.0


@dataclass
class ChunkRow:
    chunk_id: str
    source_file: str
    content: str


def fetch_source_files(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT metadata->>'source_file' FROM documents "
            "WHERE metadata->>'source_file' IS NOT NULL"
        )
        return [row[0] for row in cur.fetchall()]


def fetch_chunks_for_source(conn, source_file: str) -> list[ChunkRow]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT content, metadata->>'chunk_id' AS chunk_id,
                   metadata->>'source_file' AS source_file
            FROM documents
            WHERE metadata->>'source_file' = %s
            """,
            (source_file,),
        )
        rows = cur.fetchall()
    return [
        ChunkRow(chunk_id=r["chunk_id"], source_file=r["source_file"], content=r["content"])
        for r in rows
    ]


def sample_chunks(conn, per_doc: int, seed: int = 42) -> list[ChunkRow]:
    random.seed(seed)
    sampled: list[ChunkRow] = []
    for source_file in fetch_source_files(conn):
        chunks = fetch_chunks_for_source(conn, source_file)
        # Skip chunks too short to yield a meaningful factual question
        chunks = [c for c in chunks if len(c.content.split()) >= 40]
        if not chunks:
            continue
        k = min(per_doc, len(chunks))
        sampled.extend(random.sample(chunks, k))
    return sampled


# ---------------------------------------------------------------------------
# LLM-driven Q&A generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You write evaluation questions for a RAG system.

Given a single passage, write ONE factual question that:
- Can be answered completely and unambiguously using ONLY this passage.
- Does not reference "the passage", "the text", "the document", or similar.
- Is specific enough that a different passage would not also answer it.
- Has a concise, factually correct reference answer, using only information stated in the passage.

Respond with ONLY a JSON object, no markdown fences, no preamble:
{"question": "...", "answer": "..."}
"""


def call_llm_for_qa(client: Groq, chunk_text: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Passage:\n{chunk_text}"},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            if "question" in parsed and "answer" in parsed:
                return parsed
            print(f"  [warn] missing keys in LLM response, skipping: {raw[:120]}", file=sys.stderr)
            return None
        except json.JSONDecodeError:
            print(f"  [warn] non-JSON response, skipping chunk", file=sys.stderr)
            return None
        except Exception as e:
            # Covers Groq rate limit errors and transient network errors
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e} -- waiting {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    print("  [error] exhausted retries, skipping chunk", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a golden Q&A JSONL eval set.")
    parser.add_argument("--per-doc", type=int, default=4, help="Questions to generate per source PDF")
    parser.add_argument("--output", type=str, default="golden_qa.jsonl", help="Output JSONL path")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between LLM calls (rate limiting)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for chunk sampling")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set in environment/.env", file=sys.stderr)
        sys.exit(1)

    conn = get_connection()
    try:
        chunks = sample_chunks(conn, per_doc=args.per_doc, seed=args.seed)
    finally:
        conn.close()

    if not chunks:
        print("No eligible chunks found. Check that `documents` is populated.", file=sys.stderr)
        sys.exit(1)

    print(f"Sampled {len(chunks)} chunks across "
          f"{len({c.source_file for c in chunks})} source documents.")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    written = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks, start=1):
            print(f"[{i}/{len(chunks)}] {chunk.source_file} :: {chunk.chunk_id}")
            qa = call_llm_for_qa(client, chunk.content)
            if qa is None:
                continue

            record = {
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "contexts": [chunk.content],
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            time.sleep(args.sleep)  # stay under Groq free-tier RPM limit

    print(f"\nWrote {written}/{len(chunks)} Q&A pairs to {args.output}")


if __name__ == "__main__" and not __package__:
    main()

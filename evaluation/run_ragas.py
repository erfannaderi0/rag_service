"""
run_ragas.py

Runs the full RAG pipeline (retrieve -> generate) against the golden Q&A set
and scores it with RAGAS: faithfulness, answer relevancy, and context precision.
Results are dumped to evaluation/reports/ as both JSON (raw + aggregate) and CSV
(per-sample scores).

Note on required package versions: as of writing (Jul 2026), the current
langchain_community release (0.4.x) has dropped the vertexai chat model
submodule that ragas's LLM factory imports unconditionally, which breaks
`import ragas` on a fresh install. Pin these versions to avoid it:

    pip install "ragas==0.2.15" "langchain_community==0.2.19" \
                "langchain-groq==0.1.9" "langchain-core==0.2.43"

Usage:
    python evaluation/run_ragas.py --input evaluation/golden_qa.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and not __package__:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig

# These must match your actual app package layout.
from app.retrieve import retrieve
from app.generation.generate import call_llm
from app.generation.prompts import build_messages
from app.ingestion.embedding import Embedder  # your self-hosted BGE-M3 wrapper

load_dotenv()

DEFAULT_JUDGE_MODEL = "llama-3.1-8b-instant"  # separate Groq quota bucket from the 70b generation
# model -- also sidesteps same-model judge/generator bias as a free side effect


# ---------------------------------------------------------------------------
# Embeddings adapter: reuses your self-hosted BGE-M3 embedder so RAGAS'
# answer_relevancy metric scores in the same embedding space as retrieval,
# instead of pulling in a separate (paid) embedding API.
# ---------------------------------------------------------------------------
class BGEEmbeddingsAdapter(BaseRagasEmbeddings):
    def __init__(self):
        super().__init__()
        self._embedder = Embedder()  # adjust if your class needs constructor args

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def load_golden_set(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def unwrap_error(e: Exception) -> str:
    """tenacity's RetryError hides the real cause behind a Future repr -- unwrap it so the
    actual message from the API (which limit was hit, when it resets, token counts, etc.)
    is visible instead of just 'RetryError[<Future ... state=finished raised X>]'."""
    last_attempt = getattr(e, "last_attempt", None)
    if last_attempt is not None:
        try:
            cause = last_attempt.exception()
            if cause is not None:
                return f"{type(cause).__name__}: {cause}"
        except Exception:
            pass
    return f"{type(e).__name__}: {e}"


def run_pipeline_on_question(question: str, top_k: int = 5) -> tuple[str, list[str]]:
    """Runs the actual retrieve -> generate pipeline. Returns (answer, retrieved_context_texts)."""
    retrieved = retrieve(question, top_k=top_k)  # adjust signature if yours differs
    context_texts = [r["content"] if isinstance(r, dict) else r.content for r in retrieved]

    messages = build_messages(question, retrieved)
    answer = call_llm(messages)
    return answer, context_texts


def build_evaluation_dataset(
    golden_set: list[dict], top_k: int, pipeline_sleep: float
) -> tuple[EvaluationDataset, list[dict]]:
    samples = []
    run_records = []  # keep raw pipeline I/O alongside for the report

    for i, item in enumerate(golden_set, start=1):
        question = item["question"]
        reference = item["ground_truth"]
        print(f"[{i}/{len(golden_set)}] running pipeline: {question[:80]}")

        try:
            answer, retrieved_contexts = run_pipeline_on_question(question, top_k=top_k)
        except Exception as e:
            print(f"  [error] pipeline failed, skipping: {unwrap_error(e)}", file=sys.stderr)
            continue
        finally:
            # This loop is plain sequential Python calling call_llm() with no pacing of its
            # own -- back-to-back calls will blow through Groq's free-tier RPM limit well
            # before the golden set is exhausted (this is what caused the RateLimitErrors
            # starting partway through your last run). Sleep every iteration, success or not.
            time.sleep(pipeline_sleep)

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
        )
        run_records.append(
            {
                "question": question,
                "reference": reference,
                "response": answer,
                "retrieved_contexts": retrieved_contexts,
                "chunk_id": item.get("chunk_id"),
                "source_file": item.get("source_file"),
            }
        )

    return EvaluationDataset(samples=samples), run_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the golden Q&A set.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "golden_qa.jsonl"),
        help="Path to golden Q&A JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "reports"),
        help="Directory to write reports into",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Max concurrent LLM calls during scoring (caps concurrency, not RPM -- paired with "
             "--requests-per-second below for actual pacing).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=0.4,
        help="Sustained request rate for the RAGAS judge LLM (0.4 = ~24/min, safely under Groq's "
             "~30 RPM free-tier limit). This is what actually prevents rate-limit failures during "
             "scoring; --max-workers alone does not, since a fast-completing call frees a worker "
             "slot immediately and lets the next one fire right away.",
    )
    parser.add_argument(
        "--pipeline-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between pipeline (retrieve+generate) calls, to keep the sequential "
             "question-answering loop under Groq's RPM limit before scoring even starts.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Groq model used to judge/score. Deliberately different from your generation model "
             "(llama-3.3-70b-versatile) so it draws from a separate daily token quota -- if your "
             "generation model's TPD budget is exhausted, scoring can still proceed. Also avoids "
             "same-model judge/generator self-preference bias.",
    )
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set in environment/.env", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Golden set not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    golden_set = load_golden_set(input_path)
    print(f"Loaded {len(golden_set)} golden Q&A pairs from {input_path}")

    dataset, run_records = build_evaluation_dataset(
        golden_set, top_k=args.top_k, pipeline_sleep=args.pipeline_sleep
    )
    if len(dataset) == 0:
        print("No samples survived pipeline execution -- nothing to score.", file=sys.stderr)
        sys.exit(1)

    print(f"\nScoring {len(dataset)} samples with RAGAS "
          f"(judge model: {args.judge_model})...\n")

    # NOTE: using the same model family (Groq/Llama) as both the generation
    # backend and the RAGAS judge risks self-preference bias -- the judge may
    # rate outputs from its own model family more favorably. Swap judge_llm
    # for a different provider (e.g. Anthropic) if you want an independent
    # check; that's a one-line change here.
    #
    # A real rate limiter, not just a worker cap: max_workers only bounds how many calls run
    # concurrently, not how many happen per minute -- a fast-returning call frees its slot
    # immediately and the next one fires right away, which is how the previous run blew through
    # Groq's free-tier RPM limit even at max_workers=2. InMemoryRateLimiter enforces an actual
    # sustained requests/second ceiling regardless of how fast individual calls complete.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=args.requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=1,  # no burst allowance -- smooth, steady pacing
    )
    judge_llm = LangchainLLMWrapper(
        ChatGroq(model=args.judge_model, api_key=os.environ["GROQ_API_KEY"], rate_limiter=rate_limiter)
    )
    embeddings = BGEEmbeddingsAdapter()

    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference()]

    # max_workers still caps concurrency (so retries/backoff don't pile up chaotically), but the
    # rate_limiter above is what actually prevents hitting Groq's RPM ceiling in the first place.
    run_config = RunConfig(max_workers=args.max_workers, max_retries=15, max_wait=90, timeout=300)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,  # log per-sample failures as NaN instead of aborting the run
    )

    df = result.to_pandas()

    # Merge in the raw pipeline I/O (question/response/contexts) for traceability,
    # since `df` from ragas only guarantees the sample fields + scores.
    for col in ("chunk_id", "source_file"):
        df[col] = [r.get(col) for r in run_records[: len(df)]]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"ragas_report_{timestamp}.csv"
    json_path = output_dir / f"ragas_report_{timestamp}.json"
    latest_json_path = output_dir / "ragas_report_latest.json"

    df.to_csv(csv_path, index=False)

    aggregate_scores = {
        metric: float(df[metric].mean()) for metric in ("faithfulness", "answer_relevancy",
                                                          "llm_context_precision_with_reference")
        if metric in df.columns
    }

    report = {
        "timestamp": timestamp,
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "num_samples": len(df),
        "aggregate_scores": aggregate_scores,
        "per_sample": json.loads(df.to_json(orient="records")),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Aggregate scores ===")
    for name, score in aggregate_scores.items():
        print(f"  {name}: {score:.3f}")

    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  {latest_json_path} (always overwritten, for CI diffing)")


if __name__ == "__main__":
    main()

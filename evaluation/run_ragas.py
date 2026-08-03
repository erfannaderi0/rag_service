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
                "langchain-core==0.2.43"

Usage:
    python evaluation/run_ragas.py --input evaluation/golden_qa.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and not __package__:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import httpx

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

DEFAULT_JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"


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


_DAILY_QUOTA_RE = re.compile(r"tokens per day|TPD", re.IGNORECASE)


def is_daily_quota_error(msg: str) -> bool:
    """True if this failure is a *daily* token-budget exhaustion (TPD) rather than a
    transient per-minute/per-second rate limit. TPD errors won't clear on their own within
    a run -- pacing (sleep/rate limiters) only helps with RPM/RPS limits, not this one. Once
    you hit it, every subsequent call fails until the daily window resets, so the caller
    should stop retrying immediately instead of burning through the rest of the golden set."""
    return bool(_DAILY_QUOTA_RE.search(msg))


def load_checkpoint(path: Path) -> dict[str, dict]:
    """Load previously-completed pipeline results, keyed by question text, so a rerun after
    hitting a daily quota can resume instead of starting over from question 1."""
    if not path.exists():
        return {}
    done = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["question"]] = rec
    return done


def append_checkpoint(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline_on_question(question: str, top_k: int = 5) -> tuple[str, list[str]]:
    """Runs the actual retrieve -> generate pipeline. Returns (answer, retrieved_context_texts)."""
    retrieved = retrieve(question, top_k=top_k)  # adjust signature if yours differs
    context_texts = [r["content"] if isinstance(r, dict) else r.content for r in retrieved]

    messages = build_messages(question, retrieved)
    answer = call_llm(messages)
    return answer, context_texts


def build_evaluation_dataset(
    golden_set: list[dict], top_k: int, pipeline_sleep: float, checkpoint_path: Path
) -> tuple[EvaluationDataset, list[dict]]:
    samples = []
    run_records = []  # keep raw pipeline I/O alongside for the report

    already_done = load_checkpoint(checkpoint_path)
    if already_done:
        print(f"Resuming: {len(already_done)} question(s) already completed in a prior run "
              f"(loaded from {checkpoint_path})")

    for i, item in enumerate(golden_set, start=1):
        question = item["question"]
        reference = item["ground_truth"]

        if question in already_done:
            rec = already_done[question]
            samples.append(
                SingleTurnSample(
                    user_input=rec["question"],
                    response=rec["response"],
                    retrieved_contexts=rec["retrieved_contexts"],
                    reference=rec["reference"],
                )
            )
            run_records.append(rec)
            continue

        print(f"[{i}/{len(golden_set)}] running pipeline: {question[:80]}")

        try:
            answer, retrieved_contexts = run_pipeline_on_question(question, top_k=top_k)
        except Exception as e:
            err_msg = unwrap_error(e)
            print(f"  [error] pipeline failed, skipping: {err_msg}", file=sys.stderr)
            if is_daily_quota_error(err_msg):
                # A daily token budget won't refill mid-run no matter how long we sleep --
                # every remaining question would fail too. Stop now instead of grinding
                # through the rest of the golden set on guaranteed failures; whatever
                # succeeded so far is already saved to the checkpoint and can be resumed
                # tomorrow (or once the quota resets) by rerunning with the same --checkpoint.
                print(
                    f"  [stopping] hit a daily token-quota limit ({i - 1}/{len(golden_set)} "
                    f"done, {len(already_done)} loaded from checkpoint). Rerun later with the "
                    f"same --checkpoint path to resume from here instead of question 1.",
                    file=sys.stderr,
                )
                break
            continue
        finally:
            # Sleep still matters for the per-minute (RPM) limit -- it just can't do
            # anything about a per-day (TPD) one, which is why we also check above.
            time.sleep(pipeline_sleep)

        record = {
            "question": question,
            "reference": reference,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
            "chunk_id": item.get("chunk_id"),
            "source_file": item.get("source_file"),
        }
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
        )
        run_records.append(record)
        append_checkpoint(checkpoint_path, record)  # persist immediately, not just at the end

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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a JSONL checkpoint file for pipeline (retrieve+generate) results. If it "
             "already contains a question, that question is loaded from disk instead of "
             "re-running the pipeline -- lets you resume after hitting a daily token quota "
             "without redoing already-completed questions. Defaults to "
             "<output-dir>/pipeline_checkpoint.jsonl.",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=20,
        help="Score this many samples per RAGAS evaluate() call, writing a partial report after "
             "each batch. Keeps a timeout, crash, or Ctrl+C from losing all scoring progress -- "
             "only the in-progress batch is lost, not everything before it.",
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

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "pipeline_checkpoint.jsonl"

    dataset, run_records = build_evaluation_dataset(
        golden_set, top_k=args.top_k, pipeline_sleep=args.pipeline_sleep, checkpoint_path=checkpoint_path
    )
    if len(dataset) == 0:
        print("No samples survived pipeline execution -- nothing to score.", file=sys.stderr)
        sys.exit(1)

    print(f"\nScoring {len(dataset)} samples with RAGAS "
          f"(judge model: {args.judge_model})...\n")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=args.judge_model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # any non-empty string; Ollama ignores it
            temperature=0,
        )
    )

    embeddings = BGEEmbeddingsAdapter()

    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference()]

    run_config = RunConfig(max_workers=args.max_workers, max_retries=15, max_wait=90, timeout=300)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    partial_json_path = output_dir / f"ragas_report_{timestamp}.partial.json"

    # Score in batches rather than one evaluate() call over the whole dataset. Each metric
    # makes several sequential judge-LLM calls per sample, so a full run can take well over
    # an hour -- a timeout, crash, or Ctrl+C partway through previously meant losing every
    # score, not just the in-progress batch. Writing a partial report after each batch means
    # only the current batch's work is at risk.
    all_samples = dataset.samples
    batch_size = max(1, args.score_batch_size)
    df_batches = []

    # Warm up the model so the first real judge call doesn't race a cold model load
    httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": args.judge_model, "prompt": "hi", "stream": False},
        timeout=120,
    )
    
    for start in range(0, len(all_samples), batch_size):
        batch = all_samples[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (len(all_samples) + batch_size - 1) // batch_size
        print(f"\n--- scoring batch {batch_num}/{total_batches} ({len(batch)} samples) ---")

        result = evaluate(
            dataset=EvaluationDataset(samples=batch),
            metrics=metrics,
            llm=judge_llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,  # log per-sample failures as NaN instead of aborting the run
        )
        batch_df = result.to_pandas()
        df_batches.append(batch_df)

        # Merge in raw pipeline I/O for traceability so far, and write a partial report --
        # if the next batch fails or the run is interrupted, this file still has everything
        # scored up to this point instead of nothing at all.
        partial_df = pd.concat(df_batches, ignore_index=True)
        for col in ("chunk_id", "source_file"):
            partial_df[col] = [r.get(col) for r in run_records[: len(partial_df)]]
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "judge_model": args.judge_model,
                    "batches_completed": batch_num,
                    "total_batches": total_batches,
                    "num_samples_scored": len(partial_df),
                    "per_sample": json.loads(partial_df.to_json(orient="records")),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    df = pd.concat(df_batches, ignore_index=True)

    # Merge in the raw pipeline I/O (question/response/contexts) for traceability,
    # since `df` from ragas only guarantees the sample fields + scores.
    for col in ("chunk_id", "source_file"):
        df[col] = [r.get(col) for r in run_records[: len(df)]]

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

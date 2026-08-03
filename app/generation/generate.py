import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from groq import Groq

from app import config
from app.retrieve import retrieve
from app.generation.prompts import build_messages
# app/generation/generate.py
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils import get_logger

logger = get_logger(__name__)

# Reads GROQ_API_KEY from the environment. Loaded once, not per call.
_client = Groq()

# openai/gpt-oss-120b: switched from llama-3.3-70b-versatile, which has the
# smallest daily token budget (100K TPD) of any Groq free-tier chat model and
# was getting exhausted mid-eval-run. gpt-oss-120b doubles that to 200K TPD
# at the same 30 RPM, and being a different model family from the
# llama-3.1-8b-instant RAGAS judge, it also avoids same-model judge/generator
# self-preference bias. Recheck console.groq.com/docs/rate-limits if you
# start hitting 429s again -- limits do change.
GENERATION_MODEL = getattr(config, "GENERATION_MODEL", "openai/gpt-oss-120b")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_llm(messages: list[dict], max_tokens: int = 1024) -> str:
    """
    Takes chat-style messages (see prompt.build_messages) and returns the
    model's text answer as a plain string.

    Groq's API is OpenAI-compatible, so `messages` (including the system
    role) is passed through as-is — no need to split system out separately
    like the Anthropic Messages API requires.
    """
    kwargs = dict(
        model=GENERATION_MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
    # gpt-oss models are reasoning models: they spend hidden chain-of-thought
    # tokens before the final answer, and those count against your TPD budget
    # even though call_llm never sees them (response.choices[0].message.content
    # is only the final answer, not the reasoning). "low" keeps that internal
    # reasoning short for a straightforward extractive QA task like this one --
    # bump to "medium"/"high" only if you see answer quality drop. Ignored by
    # models that don't support it (e.g. the Llama models), so this is safe to
    # leave in even if GENERATION_MODEL changes again later.
    if GENERATION_MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"

    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def generate(query: str, top_k: int = config.TOP_K) -> dict:
    """
    Full RAG step: retrieve -> build prompt -> call LLM.
    Returns the answer plus the chunks it was grounded in, so callers
    (API layer, eval harness) can inspect/log what was actually used.
    """
    chunks = retrieve(query, top_k=top_k)
    messages = build_messages(query, chunks)
    answer = call_llm(messages)

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("Query: ")
    result = generate(query)

    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for i, c in enumerate(result["chunks"], start=1):
        print(f"  [{i}] (id={c['id']}) {c['content'][:80]!r}")

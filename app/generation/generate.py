import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from groq import Groq

from app import config
from app.retrieve import retrieve
from app.generation.prompts import build_messages

# Reads GROQ_API_KEY from the environment. Loaded once, not per call.
_client = Groq()

# llama-3.3-70b-versatile: solid quality-for-free-tier tradeoff, and the
# most generous free-tier limits of Groq's larger models (30 RPM / 12K TPM
# / 100K TPD as of writing — worth rechecking on console.groq.com if you
# start hitting 429s).
GENERATION_MODEL = getattr(config, "GENERATION_MODEL", "llama-3.3-70b-versatile")


def call_llm(messages: list[dict], max_tokens: int = 1024) -> str:
    """
    Takes chat-style messages (see prompt.build_messages) and returns the
    model's text answer as a plain string.

    Groq's API is OpenAI-compatible, so `messages` (including the system
    role) is passed through as-is — no need to split system out separately
    like the Anthropic Messages API requires.
    """
    response = _client.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
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
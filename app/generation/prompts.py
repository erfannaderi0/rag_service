SYSTEM_PROMPT = """You are a question-answering assistant. Answer the user's \
question using ONLY the information in the provided context.

Rules:
- If the context does not contain enough information to answer, say so \
explicitly instead of guessing.
- Do not use outside knowledge, even if you know the answer.
- Cite sources inline using the bracketed number of the chunk you used, \
e.g. [1], [2].
- Be concise and direct.
"""


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = ""
        metadata = chunk.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("source"):
            source = f" (source: {metadata['source']})"
        blocks.append(f"[{i}]{source}\n{chunk['content']}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Returns chat-style messages: [{"role": ..., "content": ...}, ...]
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, chunks)},
    ]


if __name__ == "__main__":
    fake_chunks = [
        {"id": 1, "content": "The sky is blue due to Rayleigh scattering.", "metadata": {"source": "physics.md"}},
        {"id": 2, "content": "Blue light scatters more than red light.", "metadata": {}},
    ]
    for m in build_messages("Why is the sky blue?", fake_chunks):
        print(f"--- {m['role']} ---\n{m['content']}\n")

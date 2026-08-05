import pytest
from tenacity import RetryError, wait_none
from unittest.mock import MagicMock

from pathlib import Path
import sys
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from app.generation import generate as generate_module


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # call_llm is decorated with @retry(..., wait=wait_exponential(...)).
    # Without this, the retry/give-up tests below would actually sleep for
    # a few seconds between attempts. monkeypatch restores the original
    # wait strategy after each test automatically.
    monkeypatch.setattr(generate_module.call_llm.retry, "wait", wait_none())


def _fake_response(text):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def test_call_llm_returns_the_models_text(monkeypatch):
    monkeypatch.setattr(
        generate_module._client.chat.completions,
        "create",
        lambda **kwargs: _fake_response("the answer"),
    )

    result = generate_module.call_llm([{"role": "user", "content": "hi"}])

    assert result == "the answer"


def test_call_llm_sets_low_reasoning_effort_for_gpt_oss_models(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(generate_module, "GENERATION_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(generate_module._client.chat.completions, "create", fake_create)

    generate_module.call_llm([{"role": "user", "content": "hi"}])

    assert captured["reasoning_effort"] == "low"


def test_call_llm_omits_reasoning_effort_for_non_gpt_oss_models(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(generate_module, "GENERATION_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setattr(generate_module._client.chat.completions, "create", fake_create)

    generate_module.call_llm([{"role": "user", "content": "hi"}])

    assert "reasoning_effort" not in captured


def test_call_llm_retries_on_failure_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate limited")
        return _fake_response("finally worked")

    monkeypatch.setattr(generate_module._client.chat.completions, "create", flaky_create)

    result = generate_module.call_llm([{"role": "user", "content": "hi"}])

    assert result == "finally worked"
    assert calls["n"] == 3


def test_call_llm_gives_up_after_max_attempts(monkeypatch):
    calls = {"n": 0}

    def always_fails(**kwargs):
        calls["n"] += 1
        raise RuntimeError("still rate limited")

    monkeypatch.setattr(generate_module._client.chat.completions, "create", always_fails)

    # @retry's default reraise=False means tenacity wraps the last failure
    # in its own RetryError rather than re-raising RuntimeError directly.
    with pytest.raises(RetryError):
        generate_module.call_llm([{"role": "user", "content": "hi"}])

    assert calls["n"] == 3  # stop_after_attempt(3)


def test_generate_wires_retrieve_prompt_and_llm_together(monkeypatch):
    fake_chunks = [{"id": 1, "content": "some fact", "metadata": {}}]
    monkeypatch.setattr(generate_module, "retrieve", lambda query, top_k: fake_chunks)
    monkeypatch.setattr(generate_module, "call_llm", lambda messages, max_tokens=1024: "final answer")

    result = generate_module.generate("what is the fact?", top_k=3)

    assert result == {
        "query": "what is the fact?",
        "answer": "final answer",
        "chunks": fake_chunks,
    }


def test_generate_passes_top_k_through_to_retrieve(monkeypatch):
    captured = {}

    def fake_retrieve(query, top_k):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(generate_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(generate_module, "call_llm", lambda messages, max_tokens=1024: "answer")

    generate_module.generate("a query", top_k=9)

    assert captured["top_k"] == 9


@pytest.mark.integration
def test_generate_against_real_groq_and_database():
    """Needs a running Postgres with ingested documents and a real
    GROQ_API_KEY. Run with: pytest --run-integration
    """
    result = generate_module.generate("What is this document about?")
    assert result["answer"]
    assert isinstance(result["chunks"], list)

"""
Shared pytest setup for the whole test suite.

Two test tiers, controlled by the `integration` marker:

- unit tests (default): fast, no external services. Postgres, Groq, and the
  BGE-M3 embedding model are all mocked/stubbed.
- integration tests (`@pytest.mark.integration`): hit the real Postgres
  instance, the real Groq API, and load the real BGE-M3 model. Skipped
  unless you pass --run-integration, since they need `docker compose up`
  for Postgres and a real GROQ_API_KEY in your environment/.env.

Run just the unit tests (fast, safe for CI):
    pytest

Run everything, including integration tests:
    pytest --run-integration
"""

import os
import sys
import types
from pathlib import Path

import pytest

# Make `app` importable regardless of where pytest is invoked from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Checked directly against argv (rather than the `config` object from
# pytest_addoption) because this module-level code runs at collection time,
# before test modules like test_retrieve.py import `app.retrieve` — and
# `app.retrieve` triggers a real BGE-M3 model load at import time via its
# module-level `_embedder = Embedder()`. The stub below has to be in place
# before that import happens.
RUN_INTEGRATION = "--run-integration" in sys.argv

# `Groq()` raises at construction time if GROQ_API_KEY isn't set at all.
# Unit tests mock the actual `.chat.completions.create()` call, so a dummy
# key is enough to let `app.generation.generate` import cleanly in
# environments (e.g. CI) that don't have a real .env.
os.environ.setdefault("GROQ_API_KEY", "unit-test-dummy-key")
os.environ.setdefault("POSTGRES_PASSWORD", "unit-test-dummy-password")

# BGEM3FlagModel downloads/loads real model weights in __init__ regardless
# of any env var, which is too slow/heavy to do on every unit test run.
# Stub the whole FlagEmbedding module so `Embedder()` becomes a no-op
# unless we're actually running integration tests.
if not RUN_INTEGRATION and "FlagEmbedding" not in sys.modules:
    stub = types.ModuleType("FlagEmbedding")

    class _StubBGEM3FlagModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            raise RuntimeError(
                "Real BGEM3FlagModel.encode() was called during a unit test. "
                "Mock Embedder.embed() (or app.retrieve._embedder) instead of "
                "letting it hit the real model."
            )

    stub.BGEM3FlagModel = _StubBGEM3FlagModel
    sys.modules["FlagEmbedding"] = stub

# TextChunker.__init__ calls tiktoken.get_encoding("cl100k_base"), which
# downloads the BPE file from openaipublic.blob.core.windows.net the first
# time it's used on a machine (afterwards it's cached locally). That's a
# hidden network dependency unit tests shouldn't rely on -- it'll work on
# your machine since the pipeline has already run and warmed the cache, but
# would break on a clean checkout or in CI. Stub it with a simple
# whitespace-based tokenizer so TextChunker's packing/overlap/hard-split
# logic can be tested deterministically, independent of tiktoken's actual
# BPE. (chunk_size/chunk_overlap in the chunking tests are chosen assuming
# 1 token == 1 whitespace-split word, since that's what this stub counts.)
import tiktoken

class _StubTiktokenEncoding:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)

tiktoken.get_encoding = lambda encoding_name: _StubTiktokenEncoding()

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help=(
            "Also run tests marked 'integration'. Needs a running Postgres "
            "(docker compose up), a real GROQ_API_KEY, and the real BGE-M3 "
            "model (slow, GPU recommended)."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: hits real Postgres/Groq/BGE-M3; skipped unless --run-integration is passed",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="needs --run-integration (real Postgres/Groq/GPU)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

import pytest

from pathlib import Path
import sys
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from app.ingestion.chunking import TextChunker
from app.ingestion.loader import PageContent


def make_page(text, page_number=1, source_file="doc.pdf"):
    return PageContent(page_number=page_number, text=text, source_file=source_file)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_short_text_produces_a_single_chunk():
    chunker = TextChunker(chunk_size=400, chunk_overlap=60)
    page = make_page("A short paragraph that easily fits in one chunk.")

    chunks = chunker.chunk_page(page, source_file="doc.pdf", doc_id="d1")

    assert len(chunks) == 1
    assert chunks[0].content == "A short paragraph that easily fits in one chunk."


def test_chunk_metadata_carries_doc_id_source_file_and_page_number():
    chunker = TextChunker(chunk_size=400, chunk_overlap=60)
    page = make_page("Some text.", page_number=3, source_file="paper.pdf")

    chunks = chunker.chunk_page(page, source_file="paper.pdf", doc_id="doc-xyz")

    assert chunks[0].metadata == {
        "doc_id": "doc-xyz",
        "source_file": "paper.pdf",
        "page_number": 3,
    }


def test_each_chunk_gets_a_unique_id():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    text = "\n\n".join(
        f"Paragraph number {i} with a little bit of extra text in it." for i in range(10)
    )
    page = make_page(text)

    chunks = chunker.chunk_page(page, source_file="doc.pdf", doc_id="d1")

    assert len(chunks) > 1
    assert len(chunks) == len({c.id for c in chunks})


def test_long_text_splits_into_multiple_chunks():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    paragraphs = [f"This is paragraph {i} with several words in it today." for i in range(8)]
    page = make_page("\n\n".join(paragraphs))

    chunks = chunker.chunk_page(page, source_file="doc.pdf", doc_id="d1")

    assert len(chunks) > 1


def test_consecutive_chunks_share_overlap_content():
    # 8 "tokens" (whitespace-split words) per paragraph. chunk_overlap=8 is
    # exactly one paragraph's worth, so the overlap loop can carry over
    # exactly the previous chunk's last paragraph -- if chunk_overlap were
    # smaller than a single paragraph's token count, no overlap could ever
    # happen at all, since paragraphs are the atomic unit overlap carries.
    chunker = TextChunker(chunk_size=17, chunk_overlap=8)
    paragraphs = [f"Paragraph {i} has some unique words here today." for i in range(6)]
    page = make_page("\n\n".join(paragraphs))

    chunks = chunker.chunk_page(page, source_file="doc.pdf", doc_id="d1")

    assert len(chunks) >= 2
    # the trailing unit of chunk N should reappear as the leading unit of
    # chunk N+1 -- that's the whole point of chunk_overlap
    tail_of_first = chunks[0].content.split("\n\n")[-1]
    head_of_second = chunks[1].content.split("\n\n")[0]
    assert tail_of_first == head_of_second


def test_oversized_single_paragraph_gets_hard_split():
    # Note: _hard_split only splits on sentence boundaries (`. `, `! `, `? `)
    # -- it has no fallback to a raw token-window split despite what its
    # docstring says. So this paragraph needs actual sentence breaks in it;
    # one long run-on sentence would NOT get split at all (see the
    # docstring-mismatch test below).
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    long_paragraph = " ".join(f"word{i}." for i in range(200))
    page = make_page(long_paragraph)

    chunks = chunker.chunk_page(page, source_file="doc.pdf", doc_id="d1")

    assert len(chunks) > 1
    # every piece should individually respect the chunk size, since a
    # single paragraph exceeding chunk_size is exactly what _hard_split exists for
    for c in chunks:
        assert chunker._n_tokens(c.content) <= chunker.chunk_size


def test_hard_split_cannot_break_a_single_run_on_sentence():
    """
    Documents current behavior rather than asserting "correct" behavior:
    _hard_split's docstring promises a "raw token-window split" fallback
    for text it can't break on sentence boundaries, but the implementation
    never does that -- it only ever splits on `. `/`! `/`? ` boundaries. A
    paragraph with no internal sentence punctuation (e.g. one giant run-on
    sentence) comes back as a single oversized, un-split unit. Flagging
    this here so it's caught explicitly if/when that fallback gets added.
    """
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    run_on_paragraph = " ".join(f"word{i}" for i in range(200)) + "."  # one period, at the very end
    page = make_page(run_on_paragraph)

    chunks = chunker.chunk_page(page, source_file="doc.pdf", doc_id="d1")

    assert len(chunks) == 1
    assert chunker._n_tokens(chunks[0].content) > chunker.chunk_size


def test_chunk_document_aggregates_chunks_across_pages():
    chunker = TextChunker(chunk_size=400, chunk_overlap=60)
    pages = [
        make_page("First page content.", page_number=1),
        make_page("Second page content.", page_number=2),
    ]

    chunks = chunker.chunk_document(pages, doc_id="doc-1")

    assert len(chunks) == 2
    assert {c.metadata["page_number"] for c in chunks} == {1, 2}
    assert all(c.metadata["doc_id"] == "doc-1" for c in chunks)


def test_chunk_document_generates_a_doc_id_when_none_given():
    chunker = TextChunker(chunk_size=400, chunk_overlap=60)
    pages = [make_page("Some content.")]

    chunks = chunker.chunk_document(pages)

    assert chunks[0].metadata["doc_id"]  # non-empty, auto-generated uuid

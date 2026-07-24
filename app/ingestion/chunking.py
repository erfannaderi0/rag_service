from __future__ import annotations

import tiktoken
import re
import uuid
from loader import PageContent
from dataclasses import dataclass, field
from loader import PageContent


@dataclass
class Chunk:
    """A chunk ready to be embedded and stored in `documents`."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    Token-aware sliding-window chunker with paragraph-boundary preference.

    Strategy:
      1. Split page text into paragraphs.
      2. Greedily pack paragraphs into chunks up to `chunk_size` tokens.
      3. If a single paragraph exceeds chunk_size, hard-split it.
      4. Apply `chunk_overlap` tokens of trailing context to the next chunk,
         so retrieval doesn't lose context across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        encoding_name: str = "cl100k_base",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def _n_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _split_paragraphs(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paras

    def _hard_split(self, text: str) -> list[str]:
        """Split an oversized paragraph on sentence boundaries first,
        falling back to a raw token-window split."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces, current = [], ""
        for s in sentences:
            candidate = f"{current} {s}".strip()
            if self._n_tokens(candidate) > self.chunk_size:
                if current:
                    pieces.append(current)
                current = s
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def chunk_page(self, page: PageContent, source_file: str, doc_id: str) -> list[Chunk]:
        paragraphs = self._split_paragraphs(page.text)

        # Expand any paragraph that alone exceeds chunk_size
        units: list[str] = []
        for p in paragraphs:
            if self._n_tokens(p) > self.chunk_size:
                units.extend(self._hard_split(p))
            else:
                units.append(p)

        chunks: list[Chunk] = []
        current_units: list[str] = []
        current_tokens = 0

        def flush():
            if not current_units:
                return
            content = "\n\n".join(current_units)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content=content,
                    metadata={
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "page_number": page.page_number,
                    },
                )
            )

        for unit in units:
            unit_tokens = self._n_tokens(unit)

            if current_tokens + unit_tokens > self.chunk_size:
                flush()
                # carry over overlap: take trailing units worth ~chunk_overlap tokens
                overlap_units, overlap_tokens = [], 0
                for u in reversed(current_units):
                    t = self._n_tokens(u)
                    if overlap_tokens + t > self.chunk_overlap:
                        break
                    overlap_units.insert(0, u)
                    overlap_tokens += t
                current_units = overlap_units
                current_tokens = overlap_tokens

            current_units.append(unit)
            current_tokens += unit_tokens

        flush()
        return chunks

    def chunk_document(self, pages: list[PageContent], doc_id: str | None = None) -> list[Chunk]:
        doc_id = doc_id or str(uuid.uuid4())
        all_chunks: list[Chunk] = []
        for page in pages:
            all_chunks.extend(
                self.chunk_page(page, source_file=page.source_file, doc_id=doc_id)
            )
        return all_chunks

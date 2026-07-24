from __future__ import annotations

from pathlib import Path
import fitz
import re
from dataclasses import dataclass


@dataclass
class PageContent:
    
    page_number: int
    text: str
    source_file: str


class PDFLoader:

    def __init__(self, min_chars_per_page: int = 20):
        #pages with less than 20 char consider empty
        self.min_chars_per_page = min_chars_per_page

    def load(self, pdf_path: str | Path) -> list[PageContent]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[PageContent] = []

        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                cleaned = self._clean_text(raw_text)

                if len(cleaned) < self.min_chars_per_page:
                    continue

                pages.append(
                    PageContent(
                        page_number=i,
                        text=cleaned,
                        source_file=pdf_path.name,
                    )
                )

        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace artifacts from PDF extraction."""
        # Strip NUL bytes — Postgres can't store them, and some PDFs embed them
        text = text.replace("\x00", "")
        # Collapse repeated newlines from PDF layout artifacts
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix hyphenated line-breaks: "informa-\ntion" -> "information"
        text = re.sub(r"-\n(?=[a-z])", "", text)
        # Collapse multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

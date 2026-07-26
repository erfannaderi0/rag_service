from __future__ import annotations

from pathlib import Path
import fitz
import re
from dataclasses import dataclass
from app.utils import clean_text


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
                cleaned = clean_text(raw_text)

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

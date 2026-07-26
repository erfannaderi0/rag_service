import logging
import re


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def clean_text(text: str) -> str:
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

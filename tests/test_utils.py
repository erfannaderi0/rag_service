from pathlib import Path
import sys
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    
from app.utils import clean_text


def test_strips_nul_bytes():
    assert clean_text("hello\x00world") == "helloworld"


def test_collapses_three_or_more_newlines_to_two():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_preserves_single_and_double_newlines():
    assert clean_text("a\nb\n\nc") == "a\nb\n\nc"


def test_fixes_hyphenated_line_breaks_before_lowercase():
    assert clean_text("informa-\ntion") == "information"


def test_leaves_hyphen_alone_before_uppercase_or_digit():
    # Only a lowercase continuation is treated as a wrapped word; this avoids
    # mangling things like "Model-\nT" or "Section-\n2" into one token.
    assert clean_text("Model-\nT") == "Model-\nT"
    assert clean_text("Section-\n2") == "Section-\n2"


def test_collapses_repeated_spaces_and_tabs():
    assert clean_text("a    b\t\tc") == "a b c"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("  \n hello world \n  ") == "hello world"


def test_empty_string_stays_empty():
    assert clean_text("") == ""


def test_combination_of_artifacts_in_one_pass():
    raw = "Header\x00\n\n\n\ninforma-\ntion   about    things\n\n\ntrailing  \n\n\n"
    assert clean_text(raw) == "Header\n\ninformation about things\n\ntrailing"

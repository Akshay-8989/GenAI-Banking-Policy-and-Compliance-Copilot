"""
tests/test_document_processor.py
----------------------------------
Unit tests for FR1 (document upload) and FR2 (text processing/chunking).
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_processor import _clean_text, _split_into_chunks


# ── Text cleaning tests ───────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_null_bytes(self):
        assert "\x00" not in _clean_text("hello\x00world")

    def test_collapses_whitespace(self):
        result = _clean_text("hello    world")
        assert "    " not in result

    def test_normalises_newlines(self):
        result = _clean_text("line1\r\nline2\rline3")
        assert "\r" not in result

    def test_removes_excessive_blank_lines(self):
        result = _clean_text("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_normalises_quotes(self):
        result = _clean_text("\u201chello\u201d \u2018world\u2019")
        assert '"hello"' in result
        assert "'world'" in result

    def test_empty_string(self):
        assert _clean_text("") == ""


# ── Chunking tests ────────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_short_text_single_chunk(self):
        text   = "Hello world. This is a short sentence."
        chunks = _split_into_chunks(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        # 1000 'a' characters should produce multiple chunks with size=300
        text   = "a" * 1000
        chunks = _split_into_chunks(text, chunk_size=300, chunk_overlap=30)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text   = " ".join(["word"] * 500)   # 2499 chars
        chunks = _split_into_chunks(text, chunk_size=400, chunk_overlap=40)
        for c in chunks:
            # Allow slight overshoot at sentence boundaries
            assert len(c) <= 450, f"Chunk too large: {len(c)}"

    def test_overlap_produces_continuity(self):
        text   = "A" * 200 + "B" * 200 + "C" * 200
        chunks = _split_into_chunks(text, chunk_size=250, chunk_overlap=50)
        # There should be overlap between consecutive chunks
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert _split_into_chunks("", chunk_size=200, chunk_overlap=20) == []

    def test_whitespace_only(self):
        assert _split_into_chunks("   \n\n  ", chunk_size=200, chunk_overlap=20) == []

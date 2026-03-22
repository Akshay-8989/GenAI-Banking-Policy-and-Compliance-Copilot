"""
document_processor.py
---------------------
FR1 – Document Upload  : accepts PDF files, saves them to UPLOAD_DIR.
FR2 – Document Processing: extracts text, cleans it, and splits it into chunks
                           ready for embedding generation.

Supports both pypdf (fast) and pdfplumber (better table/layout handling).
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single text chunk with provenance metadata."""
    text: str
    source_file: str          # original filename
    page_number: int          # 1-based
    chunk_index: int          # chunk position within page
    doc_hash: str             # SHA-256 of the source file (for dedup)
    metadata: dict = field(default_factory=dict)

    def to_langchain_doc(self):
        """Convert to a LangChain Document for use with FAISS."""
        from langchain.schema import Document
        return Document(
            page_content=self.text,
            metadata={
                "source":      self.source_file,
                "page":        self.page_number,
                "chunk_index": self.chunk_index,
                "doc_hash":    self.doc_hash,
                **self.metadata,
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_text(raw: str) -> str:
    """
    Basic text cleaning:
      - collapse excessive whitespace / newlines
      - remove null bytes and control characters
      - normalise unicode dashes and quotes
    """
    text = raw.replace("\x00", "")
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # normalise fancy punctuation
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    return text.strip()


def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Character-level sliding-window chunker.
    Tries to split on sentence boundaries first; falls back to hard cut.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # prefer splitting at a sentence boundary within the last 20% of the window
        search_start = max(start, end - chunk_size // 5)
        boundary = max(
            text.rfind(". ", search_start, end),
            text.rfind(".\n", search_start, end),
            text.rfind("\n\n",  search_start, end),
        )
        if boundary != -1:
            end = boundary + 1   # include the period / newline

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def _extract_pages_pypdf(pdf_path: Path) -> List[tuple[int, str]]:
    """Return list of (page_number, raw_text) using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def _extract_pages_pdfplumber(pdf_path: Path) -> List[tuple[int, str]]:
    """Return list of (page_number, raw_text) using pdfplumber (better for tables)."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def _extract_pages(pdf_path: Path) -> List[tuple[int, str]]:
    """Try pdfplumber first; fall back to pypdf."""
    try:
        import pdfplumber  # noqa: F401
        return _extract_pages_pdfplumber(pdf_path)
    except ImportError:
        logger.info("pdfplumber not available, using pypdf")
        return _extract_pages_pypdf(pdf_path)


# ── Public API ────────────────────────────────────────────────────────────────

def save_uploaded_file(source_path: Path, upload_dir: Path) -> Path:
    """
    Copy an uploaded PDF to UPLOAD_DIR.
    Returns the destination path.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / source_path.name
    shutil.copy2(source_path, dest)
    logger.info("Saved uploaded file → %s", dest)
    return dest


def load_and_chunk_pdf(
    pdf_path: Path,
    chunk_size: int    = 512,
    chunk_overlap: int = 64,
) -> List[DocumentChunk]:
    """
    Full pipeline: PDF → pages → clean text → chunks → DocumentChunk list.

    Args:
        pdf_path:      Path to the PDF file.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of DocumentChunk objects ready to be embedded.
    """
    logger.info("Processing PDF: %s", pdf_path.name)
    doc_hash = _file_hash(pdf_path)
    pages = _extract_pages(pdf_path)

    all_chunks: List[DocumentChunk] = []
    for page_no, raw_text in pages:
        clean = _clean_text(raw_text)
        if not clean:
            continue
        for idx, chunk_text in enumerate(_split_into_chunks(clean, chunk_size, chunk_overlap)):
            all_chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source_file=pdf_path.name,
                    page_number=page_no,
                    chunk_index=idx,
                    doc_hash=doc_hash,
                )
            )

    logger.info(
        "  → %d pages, %d chunks (chunk_size=%d, overlap=%d)",
        len(pages), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


def load_multiple_pdfs(
    pdf_paths: List[Path],
    chunk_size: int    = 512,
    chunk_overlap: int = 64,
) -> List[DocumentChunk]:
    """Convenience wrapper: process multiple PDFs and combine their chunks."""
    all_chunks: List[DocumentChunk] = []
    for p in pdf_paths:
        try:
            all_chunks.extend(load_and_chunk_pdf(p, chunk_size, chunk_overlap))
        except Exception as exc:
            logger.error("Failed to process %s: %s", p.name, exc)
    return all_chunks

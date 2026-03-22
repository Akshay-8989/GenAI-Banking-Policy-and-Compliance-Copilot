"""
document_processor.py
---------------------
Handles PDF ingestion, text extraction, cleaning, and chunking.
Supports PDF and plain-text policy documents per BRD FR1 & FR2.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    import PyPDF2
    PDF_BACKEND = "pypdf2"


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single chunk of text with full provenance metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    total_chunks: int
    word_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class ProcessedDocument:
    """Container for a fully processed document."""
    file_name: str
    file_path: str
    total_pages: int
    total_chunks: int
    chunks: List[DocumentChunk]
    raw_text: str = ""


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────

class TextCleaner:
    """Cleans extracted text from PDFs – removes noise, normalises whitespace."""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # Remove null bytes and form-feeds
        text = text.replace("\x00", " ").replace("\x0c", "\n")

        # Collapse runs of whitespace (but keep newlines meaningful)
        text = re.sub(r"[ \t]+", " ", text)

        # Reduce excessive blank lines to a single blank line
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Remove lines that are purely page numbers or headers (heuristic)
        cleaned_lines = []
        for line in text.splitlines():
            if re.match(r"^\d+$", line.strip()):   # bare page numbers
                continue
            if len(line.strip()) < 3:               # near-empty lines
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()


# ──────────────────────────────────────────────
# Text Chunker
# ──────────────────────────────────────────────

class TextChunker:
    """
    Splits text into overlapping chunks.
    Uses sentence-aware boundaries so answers stay coherent.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size          # characters
        self.chunk_overlap = chunk_overlap    # characters

    def split(self, text: str) -> List[str]:
        """Return a list of overlapping text chunks."""
        if not text:
            return []

        # Split on sentence boundaries first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from previous
                if len(current) > self.chunk_overlap:
                    overlap_text = current[-self.chunk_overlap:]
                else:
                    overlap_text = current
                current = (overlap_text + " " + sentence).strip()

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c.strip()) > 30]


# ──────────────────────────────────────────────
# PDF Reader
# ──────────────────────────────────────────────

class PDFReader:
    """Reads PDF files and returns per-page text."""

    def read(self, file_path: str) -> List[dict]:
        """Returns list of {page_number, text} dicts."""
        pages = []
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if PDF_BACKEND == "pdfplumber":
            pages = self._read_pdfplumber(file_path)
        else:
            pages = self._read_pypdf2(file_path)

        logger.info(f"Read {len(pages)} pages from '{path.name}' using {PDF_BACKEND}")
        return pages

    def _read_pdfplumber(self, file_path: str) -> List[dict]:
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page_number": i, "text": text})
        return pages

    def _read_pypdf2(self, file_path: str) -> List[dict]:
        import PyPDF2
        pages = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page_number": i, "text": text})
        return pages


class TextFileReader:
    """Reads plain .txt documents."""

    def read(self, file_path: str) -> List[dict]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [{"page_number": 1, "text": content}]


# ──────────────────────────────────────────────
# Document Processor (orchestrator)
# ──────────────────────────────────────────────

class DocumentProcessor:
    """
    Main pipeline: load → clean → chunk → return DocumentChunks.
    Covers BRD FR1 (upload) and FR2 (processing).
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.pdf_reader = PDFReader()
        self.txt_reader = TextFileReader()

    def process_file(self, file_path: str) -> ProcessedDocument:
        """Process a single document file into chunks."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}")

        logger.info(f"Processing document: {path.name}")

        # Read pages
        if ext == ".pdf":
            pages = self.pdf_reader.read(file_path)
        else:
            pages = self.txt_reader.read(file_path)

        # Clean + chunk per page, track provenance
        all_chunks: List[DocumentChunk] = []
        full_raw_text = ""

        chunk_global_idx = 0
        for page_data in pages:
            cleaned = self.cleaner.clean(page_data["text"])
            full_raw_text += cleaned + "\n\n"

            page_chunks = self.chunker.split(cleaned)
            for local_idx, chunk_text in enumerate(page_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{path.stem}_p{page_data['page_number']}_c{local_idx}",
                    text=chunk_text,
                    source_file=path.name,
                    page_number=page_data["page_number"],
                    chunk_index=chunk_global_idx,
                    total_chunks=0,   # patched below
                    metadata={
                        "file_path": str(file_path),
                        "file_name": path.name,
                        "page": page_data["page_number"],
                    },
                )
                all_chunks.append(chunk)
                chunk_global_idx += 1

        # Patch total_chunks
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total

        doc = ProcessedDocument(
            file_name=path.name,
            file_path=str(file_path),
            total_pages=len(pages),
            total_chunks=total,
            chunks=all_chunks,
            raw_text=full_raw_text,
        )

        logger.success(f"'{path.name}' → {total} chunks from {len(pages)} pages")
        return doc

    def process_directory(self, dir_path: str) -> List[ProcessedDocument]:
        """Process all supported documents in a directory."""
        directory = Path(dir_path)
        docs = []
        files = [f for f in directory.iterdir() if f.suffix.lower() in self.SUPPORTED_EXTENSIONS]

        if not files:
            logger.warning(f"No supported documents found in '{dir_path}'")
            return docs

        for file in files:
            try:
                doc = self.process_file(str(file))
                docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to process '{file.name}': {e}")

        logger.info(f"Processed {len(docs)}/{len(files)} documents from '{dir_path}'")
        return docs

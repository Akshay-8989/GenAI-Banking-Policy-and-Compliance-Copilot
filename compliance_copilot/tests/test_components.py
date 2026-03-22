"""
tests/test_components.py
------------------------
Unit tests for document processor, embeddings, vector store, and RAG pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from document_processor import (
    TextCleaner, TextChunker, DocumentProcessor, DocumentChunk
)
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore, SearchResult


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_text():
    return """
    Know Your Customer (KYC) Policy – Section 3.1
    
    All new customers must provide the following documents during onboarding:
    1. Government-issued photo ID (passport or national ID card)
    2. Proof of address dated within the last 3 months (utility bill or bank statement)
    3. Tax identification number (PAN card for Indian residents)
    
    High-risk customers require enhanced due diligence as per RBI circular RBI/2019-20/150.
    All KYC documents must be verified against original copies and retained for a minimum
    of 5 years after the end of the customer relationship.
    
    Anti-Money Laundering (AML) Reporting
    Transactions exceeding INR 10 lakhs in a single day must be reported to FIU-IND
    within 7 working days. Suspicious transaction reports (STR) must be filed within
    24 hours of detection regardless of transaction amount.
    """


@pytest.fixture
def sample_txt_file(sample_text, tmp_path):
    f = tmp_path / "test_policy.txt"
    f.write_text(sample_text)
    return str(f)


@pytest.fixture
def vector_store_dir(tmp_path):
    return str(tmp_path / "vector_store")


# ──────────────────────────────────────────────
# TextCleaner Tests
# ──────────────────────────────────────────────

class TestTextCleaner:
    def test_removes_null_bytes(self):
        cleaner = TextCleaner()
        assert "\x00" not in cleaner.clean("hello\x00world")

    def test_collapses_whitespace(self):
        cleaner = TextCleaner()
        result = cleaner.clean("hello   world")
        assert "  " not in result

    def test_handles_empty_input(self):
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_removes_bare_page_numbers(self, sample_text):
        cleaner = TextCleaner()
        text_with_numbers = "Policy Section 1\n\n3\n\nSome content here."
        result = cleaner.clean(text_with_numbers)
        lines = result.splitlines()
        assert "3" not in lines


# ──────────────────────────────────────────────
# TextChunker Tests
# ──────────────────────────────────────────────

class TestTextChunker:
    def test_produces_chunks(self, sample_text):
        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.split(sample_text)
        assert len(chunks) > 0

    def test_chunk_size_respected(self, sample_text):
        chunk_size = 300
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=50)
        chunks = chunker.split(sample_text)
        # Most chunks should be near the chunk size (some may be shorter)
        oversized = [c for c in chunks if len(c) > chunk_size * 1.5]
        assert len(oversized) == 0, f"Oversized chunks found: {[len(c) for c in oversized]}"

    def test_handles_empty_text(self):
        chunker = TextChunker()
        assert chunker.split("") == []

    def test_no_tiny_chunks(self, sample_text):
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.split(sample_text)
        tiny = [c for c in chunks if len(c) < 30]
        assert len(tiny) == 0


# ──────────────────────────────────────────────
# DocumentProcessor Tests
# ──────────────────────────────────────────────

class TestDocumentProcessor:
    def test_process_txt_file(self, sample_txt_file):
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
        doc = processor.process_file(sample_txt_file)

        assert doc.file_name == "test_policy.txt"
        assert doc.total_chunks > 0
        assert len(doc.chunks) == doc.total_chunks
        assert all(isinstance(c, DocumentChunk) for c in doc.chunks)

    def test_chunk_provenance(self, sample_txt_file):
        processor = DocumentProcessor()
        doc = processor.process_file(sample_txt_file)

        for chunk in doc.chunks:
            assert chunk.source_file == "test_policy.txt"
            assert chunk.page_number >= 1
            assert chunk.text.strip() != ""
            assert chunk.chunk_id != ""

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "test.docx"
        bad_file.write_text("content")
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported"):
            processor.process_file(str(bad_file))

    def test_process_directory_empty(self, tmp_path):
        processor = DocumentProcessor()
        results = processor.process_directory(str(tmp_path))
        assert results == []

    def test_process_directory_with_files(self, sample_text, tmp_path):
        for name in ["policy1.txt", "policy2.txt"]:
            (tmp_path / name).write_text(sample_text)

        processor = DocumentProcessor(chunk_size=300)
        docs = processor.process_directory(str(tmp_path))
        assert len(docs) == 2


# ──────────────────────────────────────────────
# EmbeddingModel Tests
# ──────────────────────────────────────────────

class TestEmbeddingModel:
    @pytest.fixture(autouse=True)
    def model(self):
        self.model = EmbeddingModel("all-MiniLM-L6-v2")

    def test_embed_texts_shape(self):
        texts = ["KYC policy", "AML reporting threshold"]
        vecs = self.model.embed_texts(texts, show_progress=False)
        assert vecs.shape[0] == 2
        assert vecs.shape[1] == self.model.get_dimension()

    def test_embed_empty(self):
        vecs = self.model.embed_texts([])
        assert vecs.shape[0] == 0

    def test_embed_query_shape(self):
        vec = self.model.embed_query("What is KYC?")
        assert vec.shape == (1, self.model.get_dimension())

    def test_normalised_vectors(self):
        """Embeddings should be L2-normalised (dot product ≈ 1 with itself)."""
        vec = self.model.embed_query("test")
        norm = np.linalg.norm(vec[0])
        assert abs(norm - 1.0) < 0.01


# ──────────────────────────────────────────────
# FAISSVectorStore Tests
# ──────────────────────────────────────────────

class TestFAISSVectorStore:
    @pytest.fixture(autouse=True)
    def setup(self, vector_store_dir, sample_text):
        self.store_dir = vector_store_dir
        self.store = FAISSVectorStore(vector_store_dir, embedding_dim=384)
        self.embedding_dim = 384

        # Create dummy chunks and embeddings
        self.chunks = [
            DocumentChunk(
                chunk_id=f"doc_p1_c{i}",
                text=f"Policy chunk {i}: {sample_text[:100]}",
                source_file="test_policy.txt",
                page_number=1,
                chunk_index=i,
                total_chunks=3,
            )
            for i in range(3)
        ]
        # L2-normalised random embeddings
        raw = np.random.randn(3, self.embedding_dim).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.embeddings = raw / norms

    def test_add_and_search(self):
        self.store.add_chunks(self.chunks, self.embeddings)
        query = self.embeddings[0:1]   # use first chunk as its own query
        results = self.store.search(query, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= results[1].score   # ordered by score

    def test_save_and_load(self):
        self.store.add_chunks(self.chunks, self.embeddings)
        self.store.save()

        new_store = FAISSVectorStore(self.store_dir)
        loaded = new_store.load()

        assert loaded is True
        assert len(new_store._chunks) == 3

    def test_search_empty_store(self):
        results = self.store.search(self.embeddings[0:1])
        assert results == []

    def test_get_stats(self):
        self.store.add_chunks(self.chunks, self.embeddings, "test_policy.txt")
        stats = self.store.get_stats()
        assert stats["total_chunks"] == 3
        assert "test_policy.txt" in stats["documents"]

    def test_clear(self):
        self.store.add_chunks(self.chunks, self.embeddings)
        self.store.clear()
        assert self.store.get_stats()["total_chunks"] == 0


# ──────────────────────────────────────────────
# Integration: RAGPipeline Smoke Test
# ──────────────────────────────────────────────

class TestRAGPipelineIntegration:
    def test_ingest_and_query(self, sample_txt_file, tmp_path):
        """End-to-end: ingest a text file and query it."""
        from rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            vector_store_path=str(tmp_path / "vs"),
            embedding_model="all-MiniLM-L6-v2",
            llm_model="microsoft/phi-2",   # will use stub if not available
            chunk_size=300,
            chunk_overlap=50,
            top_k=2,
        )

        # Ingest
        result = pipeline.ingest_file(sample_txt_file)
        assert result["status"] == "success"
        assert result["chunks"] > 0

        # Query
        response = pipeline.query("What documents are required for KYC?")
        assert response.status in ("success", "no_context")
        assert len(response.answer) > 0
        if response.status == "success":
            assert len(response.citations) > 0
            assert response.citations[0].document == "test_policy.txt"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

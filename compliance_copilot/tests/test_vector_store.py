"""
tests/test_vector_store.py
--------------------------
Unit tests for FR3 (embeddings) and FR4 (FAISS storage).
Uses a temporary directory so the real vectorstore is never touched.
Run with:  pytest tests/ -v
"""

import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_chunks(n: int = 5):
    """Generate synthetic DocumentChunk objects for testing."""
    from src.document_processor import DocumentChunk
    return [
        DocumentChunk(
            text=f"The KYC policy requires document verification step {i}. "
                 f"Customers must provide a government-issued photo ID.",
            source_file=f"kyc_policy.pdf",
            page_number=i + 1,
            chunk_index=0,
            doc_hash="abc123",
        )
        for i in range(n)
    ]


class TestComplianceVectorStore:
    def _make_store(self, tmp_path):
        from src.vector_store import ComplianceVectorStore
        return ComplianceVectorStore(
            persist_dir=tmp_path,
            embedding_model="all-MiniLM-L6-v2",
        )

    def test_empty_store_returns_zero(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.document_count() == 0

    def test_add_documents_increases_count(self, tmp_path):
        store  = self._make_store(tmp_path)
        chunks = _make_chunks(5)
        store.add_documents(chunks)
        assert store.document_count() == 5

    def test_similarity_search_returns_results(self, tmp_path):
        store  = self._make_store(tmp_path)
        chunks = _make_chunks(10)
        store.add_documents(chunks)
        results = store.similarity_search("KYC document requirements", k=3)
        assert len(results) == 3
        for doc, score in results:
            assert 0.0 <= score <= 1.0

    def test_similarity_search_relevance_ordering(self, tmp_path):
        store  = self._make_store(tmp_path)
        chunks = _make_chunks(10)
        store.add_documents(chunks)
        results = store.similarity_search("KYC document requirements", k=4)
        scores  = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results should be highest-score first"

    def test_reset_clears_store(self, tmp_path):
        store  = self._make_store(tmp_path)
        chunks = _make_chunks(5)
        store.add_documents(chunks)
        assert store.document_count() > 0
        store.reset()
        assert store.document_count() == 0

    def test_persistence_across_instances(self, tmp_path):
        """Index saved by one instance should be loadable by another."""
        store1 = self._make_store(tmp_path)
        store1.add_documents(_make_chunks(5))

        store2 = self._make_store(tmp_path)   # new instance, same directory
        assert store2.document_count() == 5

    def test_add_documents_empty_list(self, tmp_path):
        store = self._make_store(tmp_path)
        # Should not raise
        store.add_documents([])
        assert store.document_count() == 0

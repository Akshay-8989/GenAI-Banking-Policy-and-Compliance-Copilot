"""
tests/test_rag_pipeline.py
--------------------------
Integration tests for the full RAG pipeline (FR5-FR8).
The LLM is mocked so tests run without loading Phi-2.
Run with:  pytest tests/ -v
"""

import sys, tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _seed_vectorstore(store, n=8):
    from src.document_processor import DocumentChunk
    chunks = [
        DocumentChunk(
            text=(
                "KYC (Know Your Customer) policy requires all new customers to submit: "
                "1) Valid government-issued photo ID (passport or national ID). "
                "2) Proof of address not older than 3 months. "
                "3) Source of funds declaration for accounts above $10,000."
            ),
            source_file="kyc_policy.pdf",
            page_number=i + 1,
            chunk_index=0,
            doc_hash="seed123",
        )
        for i in range(n)
    ]
    store.add_documents(chunks)
    return chunks


class TestRAGPipelineIntegration:
    def _make_pipeline(self, tmp_path):
        from src.rag_pipeline import ComplianceRAGPipeline
        from src.vector_store import ComplianceVectorStore
        vs = ComplianceVectorStore(tmp_path, "all-MiniLM-L6-v2")
        pipeline = ComplianceRAGPipeline(vectorstore=vs, top_k=3)
        return pipeline, vs

    def test_empty_store_returns_warning(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        response = pipeline.query("What are KYC requirements?")
        assert not response.has_context
        assert "upload" in response.answer.lower() or "no" in response.answer.lower()

    def test_blank_question_handled(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        response = pipeline.query("   ")
        assert response.answer  # should return some message

    def test_query_returns_sources(self, tmp_path):
        pipeline, vs = self._make_pipeline(tmp_path)
        _seed_vectorstore(vs)
        # Mock the LLM so we don't load Phi-2 in tests
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "KYC requires a valid photo ID and proof of address."
        pipeline._llm = mock_llm
        response = pipeline.query("What documents are required for KYC?")
        assert response.has_context
        assert len(response.sources) > 0
        assert len(response.sources) <= 3

    def test_sources_have_required_fields(self, tmp_path):
        pipeline, vs = self._make_pipeline(tmp_path)
        _seed_vectorstore(vs)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Photo ID and address proof are required."
        pipeline._llm = mock_llm
        response = pipeline.query("What is needed for KYC?")
        for src in response.sources:
            assert src.document_name
            assert src.page_number >= 1
            assert src.excerpt
            assert 0.0 <= src.relevance_score <= 1.0

    def test_format_sources_output(self, tmp_path):
        pipeline, vs = self._make_pipeline(tmp_path)
        _seed_vectorstore(vs)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Some answer."
        pipeline._llm = mock_llm
        response  = pipeline.query("What is needed for KYC?")
        formatted = response.format_sources()
        assert "kyc_policy.pdf" in formatted
        assert "Page" in formatted

    def test_response_time_target(self, tmp_path):
        """FR: response time < 5 seconds (excluding model loading)."""
        import time
        pipeline, vs = self._make_pipeline(tmp_path)
        _seed_vectorstore(vs)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Fast answer."
        pipeline._llm = mock_llm
        t0       = time.time()
        response = pipeline.query("What is the AML reporting threshold?")
        elapsed  = time.time() - t0
        assert elapsed < 5.0, f"Response took {elapsed:.2f}s (target < 5s)"

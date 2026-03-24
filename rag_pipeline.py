"""
rag_pipeline.py
---------------
FR5  – Question Answering Interface
FR6  – Context Retrieval
FR7  – AI Response Generation
FR8  – Source Citation

Orchestrates the full RAG flow:
  user question
      → embed query
      → FAISS similarity search
      → format retrieved chunks as context
      → Phi-2 generates answer
      → return answer + source citations
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import (
    EMBEDDING_MODEL,
    MAX_NEW_TOKENS,
    PHI2_MODEL_NAME,
    TEMPERATURE,
    TOP_K_RESULTS,
    VECTORSTORE_DIR,
)
from src.vector_store import ComplianceVectorStore

logger = logging.getLogger(__name__)


# ── Response Model ─────────────────────────────────────────────────────────────

@dataclass
class CitedSource:
    """FR8 – one source citation attached to an answer."""
    document_name: str
    page_number:   int
    excerpt:       str          # short snippet from the chunk
    relevance_score: float


@dataclass
class RAGResponse:
    """Full response returned to the UI."""
    question:    str
    answer:      str
    sources:     List[CitedSource] = field(default_factory=list)
    has_context: bool = True    # False if vector store was empty

    def format_sources(self) -> str:
        """Human-readable source list for display."""
        if not self.sources:
            return "No sources found."
        lines = []
        for i, s in enumerate(self.sources, start=1):
            lines.append(
                f"[{i}] 📄 {s.document_name}  |  Page {s.page_number}  "
                f"|  Relevance: {s.relevance_score:.0%}\n"
                f"    …{s.excerpt}…"
            )
        return "\n".join(lines)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class ComplianceRAGPipeline:
    """
    End-to-end RAG pipeline for the Banking Policy Compliance Copilot.

    Args:
        vectorstore:    ComplianceVectorStore instance (already populated).
        top_k:          Number of chunks to retrieve per query.
        model_name:     Phi-2 (or any causal-LM) HuggingFace ID.
        max_new_tokens: Max tokens for LLM generation.
        temperature:    Generation temperature.
    """

    def __init__(
        self,
        vectorstore:    Optional[ComplianceVectorStore] = None,
        top_k:          int   = TOP_K_RESULTS,
        model_name:     str   = "google/flan-t5-base",
        max_new_tokens: int   = 256,
        temperature:    float = TEMPERATURE,
    ):
        self.top_k          = top_k
        self.model_name     = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        # Vector store
        self.vectorstore = vectorstore or ComplianceVectorStore(persist_dir=VECTORSTORE_DIR)

        # LLM is loaded lazily (first query) to avoid blocking app startup
        self._llm = None

    # ── Lazy LLM loader ───────────────────────────────────────────────────────

    def _get_llm(self):
        if self._llm is None:
            from src.llm_engine import get_llm
            self._llm = get_llm(
                model_name     = self.model_name,
                max_new_tokens = self.max_new_tokens,
                temperature    = self.temperature,
            )
        return self._llm

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline for a user question.

        Steps:
            1. Retrieve top-k relevant chunks from FAISS
            2. Build context string from chunks
            3. Generate answer with Phi-2
            4. Package answer + citations into RAGResponse

        Args:
            question: Natural language question from the user.

        Returns:
            RAGResponse with answer text and cited sources.
        """
        question = question.strip()
        if not question:
            return RAGResponse(
                question=question,
                answer="Please enter a question.",
                has_context=False,
            )

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        try:
            results = self.vectorstore.similarity_search(question, k=self.top_k)
        except RuntimeError as exc:
            logger.warning("Vector store empty or error: %s", exc)
            return RAGResponse(
                question=question,
                answer=(
                    "⚠️ No policy documents have been indexed yet. "
                    "Please upload PDF documents first using the sidebar."
                ),
                has_context=False,
            )

        if not results:
            return RAGResponse(
                question=question,
                answer="I could not find any relevant information in the policy documents.",
                sources=[],
            )

        # ── Step 2: Build context ─────────────────────────────────────────────
        context_parts = []
        sources: List[CitedSource] = []

        for rank, (doc, score) in enumerate(results, start=1):
            meta    = doc.metadata
            excerpt = textwrap.shorten(doc.page_content, width=200, placeholder="...")

            context_parts.append(
                f"[Source {rank}: {meta.get('source','unknown')},"
                f" Page {meta.get('page', '?')}]\n{doc.page_content}"
            )
            sources.append(CitedSource(
                document_name   = meta.get("source", "Unknown Document"),
                page_number     = meta.get("page",   0),
                excerpt         = excerpt,
                relevance_score = score,
            ))

        context = "\n\n---\n\n".join(context_parts)

        # ── Step 3: Generate answer ───────────────────────────────────────────
        from src.llm_engine import build_prompt
        prompt = build_prompt(context=context, question=question)

        logger.info("Sending prompt to Phi-2 (%d chars) …", len(prompt))
        try:
            llm    = self._get_llm()
            answer = llm.generate(prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = (
                "⚠️ The language model encountered an error while generating an answer. "
                f"Technical details: {exc}"
            )

        # ── Step 4: Return ────────────────────────────────────────────────────
        return RAGResponse(
            question    = question,
            answer      = answer,
            sources     = sources,
            has_context = True,
        )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path, chunk_size: int = None, chunk_overlap: int = None):
        """
        Ingest a single PDF into the vector store.
        Useful for programmatic use outside the UI.
        """
        from pathlib import Path
        from src.config import CHUNK_OVERLAP, CHUNK_SIZE
        from src.document_processor import load_and_chunk_pdf

        pdf_path = Path(pdf_path)
        cs  = chunk_size    or CHUNK_SIZE
        co  = chunk_overlap or CHUNK_OVERLAP
        chunks = load_and_chunk_pdf(pdf_path, chunk_size=cs, chunk_overlap=co)
        self.vectorstore.add_documents(chunks)
        logger.info("Ingested %s → %d chunks", pdf_path.name, len(chunks))
        return len(chunks)

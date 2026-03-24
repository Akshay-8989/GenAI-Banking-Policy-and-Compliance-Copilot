"""
rag_pipeline.py  –  Full RAG orchestration with visible error reporting
"""

from __future__ import annotations
import logging, textwrap, traceback
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import (
    EMBEDDING_MODEL, MAX_NEW_TOKENS, PHI2_MODEL_NAME,
    TEMPERATURE, TOP_K_RESULTS, VECTORSTORE_DIR,
)
from src.vector_store import ComplianceVectorStore

logger = logging.getLogger(__name__)


@dataclass
class CitedSource:
    document_name:   str
    page_number:     int
    excerpt:         str
    relevance_score: float


@dataclass
class RAGResponse:
    question:    str
    answer:      str
    sources:     List[CitedSource] = field(default_factory=list)
    has_context: bool = True
    error:       str  = ""        # non-empty if something went wrong

    def format_sources(self) -> str:
        if not self.sources:
            return "No sources found."
        lines = []
        for i, s in enumerate(self.sources, 1):
            lines.append(
                f"[{i}] 📄 {s.document_name}  |  Page {s.page_number}  "
                f"|  Relevance: {s.relevance_score:.0%}\n"
                f"    …{s.excerpt}…"
            )
        return "\n".join(lines)


class ComplianceRAGPipeline:

    def __init__(
        self,
        vectorstore:    Optional[ComplianceVectorStore] = None,
        top_k:          int   = TOP_K_RESULTS,
        model_name:     str   = PHI2_MODEL_NAME,
        max_new_tokens: int   = MAX_NEW_TOKENS,
        temperature:    float = TEMPERATURE,
    ):
        self.top_k          = top_k
        self.model_name     = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.vectorstore    = vectorstore or ComplianceVectorStore(persist_dir=VECTORSTORE_DIR)
        self._llm           = None
        self._llm_error     = ""    # store load error so we show it once

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self._llm_error:
            raise RuntimeError(self._llm_error)
        try:
            from src.llm_engine import get_llm
            self._llm = get_llm(
                model_name     = self.model_name,
                max_new_tokens = self.max_new_tokens,
                temperature    = self.temperature,
            )
            return self._llm
        except Exception as exc:
            self._llm_error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            raise RuntimeError(self._llm_error)

    # ── Fallback: return raw retrieved chunks as the answer ───────────────────
    def _fallback_answer(self, question: str, results) -> str:
        """
        When Phi-2 can't load, return the most relevant retrieved text directly.
        This is still useful — the user can read the exact policy text.
        """
        lines = [
            "⚠️ **Language model unavailable** — showing raw policy excerpts instead:\n"
        ]
        for i, (doc, score) in enumerate(results, 1):
            lines.append(
                f"**Excerpt {i}** (Page {doc.metadata.get('page','?')}, "
                f"relevance {score:.0%}):\n"
                f"> {doc.page_content.strip()}\n"
            )
        return "\n".join(lines)

    def query(self, question: str) -> RAGResponse:
        question = question.strip()
        if not question:
            return RAGResponse(question=question, answer="Please enter a question.",
                               has_context=False)

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        try:
            results = self.vectorstore.similarity_search(question, k=self.top_k)
        except RuntimeError as exc:
            return RAGResponse(
                question=question, has_context=False,
                answer="⚠️ No documents indexed yet. Upload a PDF first.",
                error=str(exc),
            )

        if not results:
            return RAGResponse(
                question=question, sources=[],
                answer="I could not find any relevant information in the policy documents.",
            )

        # ── Step 2: Build context + sources ───────────────────────────────────
        context_parts, sources = [], []
        for rank, (doc, score) in enumerate(results, 1):
            meta    = doc.metadata
            excerpt = textwrap.shorten(doc.page_content, width=200, placeholder="...")
            context_parts.append(
                f"[Source {rank}: {meta.get('source','unknown')},"
                f" Page {meta.get('page','?')}]\n{doc.page_content}"
            )
            sources.append(CitedSource(
                document_name   = meta.get("source", "Unknown"),
                page_number     = meta.get("page", 0),
                excerpt         = excerpt,
                relevance_score = score,
            ))
        context = "\n\n---\n\n".join(context_parts)

        # ── Step 3: Generate answer with LLM ──────────────────────────────────
        try:
            from src.llm_engine import build_prompt
            prompt = build_prompt(context=context, question=question)
            logger.info("Sending prompt to LLM (%d chars)…", len(prompt))
            llm    = self._get_llm()
            answer = llm.generate(prompt)
            if not answer or not answer.strip():
                answer = self._fallback_answer(question, results)
        except Exception as exc:
            full_error = traceback.format_exc()
            logger.error("LLM failed:\n%s", full_error)
            # Show fallback but also surface the real error
            answer = self._fallback_answer(question, results)
            return RAGResponse(
                question=question, answer=answer, sources=sources,
                has_context=True,
                error=f"**LLM Error:** `{type(exc).__name__}: {exc}`\n\n"
                      f"Showing raw document excerpts instead.",
            )

        return RAGResponse(question=question, answer=answer,
                           sources=sources, has_context=True)

    def ingest_pdf(self, pdf_path, chunk_size=None, chunk_overlap=None):
        from pathlib import Path
        from src.config import CHUNK_OVERLAP, CHUNK_SIZE
        from src.document_processor import load_and_chunk_pdf
        pdf_path = Path(pdf_path)
        chunks   = load_and_chunk_pdf(pdf_path,
                                      chunk_size    = chunk_size    or CHUNK_SIZE,
                                      chunk_overlap = chunk_overlap or CHUNK_OVERLAP)
        self.vectorstore.add_documents(chunks)
        return len(chunks)

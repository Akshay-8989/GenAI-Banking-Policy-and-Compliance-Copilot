"""
rag_pipeline.py
---------------
Orchestrates the full RAG pipeline:
  Document ingestion → Embedding → FAISS storage → Retrieval → Phi-2 generation
Covers BRD FR5–FR8 (query interface, retrieval, generation, citations).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger

from document_processor import DocumentProcessor, DocumentChunk
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore, SearchResult
from llm_phi2 import Phi2LLM


# ──────────────────────────────────────────────
# Response Model (with citations)
# ──────────────────────────────────────────────

@dataclass
class Citation:
    """Source reference for a retrieved context chunk – BRD FR8."""
    rank: int
    document: str
    page_number: int
    relevance_score: float
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "document": self.document,
            "page_number": self.page_number,
            "relevance_score": round(self.relevance_score, 4),
            "excerpt": self.excerpt[:300] + ("…" if len(self.excerpt) > 300 else ""),
        }


@dataclass
class RAGResponse:
    """Full answer with citations returned to the UI / API."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: int = 0
    status: str = "success"   # success | no_context | error
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunks": self.retrieved_chunks,
            "status": self.status,
            "error_message": self.error_message,
        }


# ──────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.

    Usage:
        pipeline = RAGPipeline.from_config()
        pipeline.ingest_directory("./data/policies")
        response = pipeline.query("What are KYC document requirements?")
    """

    def __init__(
        self,
        vector_store_path: str = "./data/vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "microsoft/phi-2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 4,
        llm_max_tokens: int = 512,
        llm_temperature: float = 0.1,
        llm_device: str = "cpu",
    ):
        self.top_k = top_k

        # Initialise components
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedder = EmbeddingModel(embedding_model)
        self.vector_store = FAISSVectorStore(vector_store_path)
        self.llm = Phi2LLM(
            model_name=llm_model,
            max_new_tokens=llm_max_tokens,
            temperature=llm_temperature,
            device=llm_device,
        )

        # Try to load existing index
        self.vector_store.load()
        logger.info("RAG pipeline ready.")

    @classmethod
    def from_env(cls) -> "RAGPipeline":
        """Build pipeline from environment variables / .env file."""
        from dotenv import load_dotenv
        load_dotenv()

        return cls(
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_MODEL_NAME", "microsoft/phi-2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
            top_k=int(os.getenv("TOP_K_RESULTS", 4)),
            llm_max_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", 512)),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            llm_device=os.getenv("LLM_DEVICE", "cpu"),
        )

    # ── Ingestion ──────────────────────────────

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document into the vector store."""
        logger.info(f"Ingesting: {file_path}")

        # Process
        doc = self.processor.process_file(file_path)
        if not doc.chunks:
            return {"status": "skipped", "reason": "No text extracted", "file": file_path}

        # Embed
        embeddings = self.embedder.embed_chunks(doc.chunks)

        # Store
        self.vector_store.add_chunks(doc.chunks, embeddings, doc.file_name)
        self.vector_store.save()

        return {
            "status": "success",
            "file": doc.file_name,
            "pages": doc.total_pages,
            "chunks": doc.total_chunks,
        }

    def ingest_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Ingest all documents in a directory."""
        results = []
        docs = self.processor.process_directory(dir_path)

        for doc in docs:
            try:
                embeddings = self.embedder.embed_chunks(doc.chunks)
                self.vector_store.add_chunks(doc.chunks, embeddings, doc.file_name)
                results.append({
                    "status": "success",
                    "file": doc.file_name,
                    "pages": doc.total_pages,
                    "chunks": doc.total_chunks,
                })
                logger.success(f"Ingested: {doc.file_name} ({doc.total_chunks} chunks)")
            except Exception as e:
                logger.error(f"Failed to ingest {doc.file_name}: {e}")
                results.append({"status": "error", "file": doc.file_name, "error": str(e)})

        if docs:
            self.vector_store.save()

        return results

    # ── Query ──────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Full RAG query:
          1. Embed question
          2. Retrieve top-k chunks
          3. Generate answer with Phi-2
          4. Return answer + citations
        """
        if not question.strip():
            return RAGResponse(
                question=question,
                answer="Please enter a valid question.",
                status="error",
                error_message="Empty question",
            )

        stats = self.vector_store.get_stats()
        if stats["total_chunks"] == 0:
            return RAGResponse(
                question=question,
                answer="No policy documents have been ingested yet. Please upload documents first.",
                status="no_context",
            )

        try:
            # 1. Embed the question
            query_vec = self.embedder.embed_query(question)

            # 2. Retrieve relevant chunks
            search_results: List[SearchResult] = self.vector_store.search(
                query_vec, top_k=self.top_k
            )

            if not search_results:
                return RAGResponse(
                    question=question,
                    answer="No relevant policy information found for this question.",
                    status="no_context",
                )

            # 3. Build context for LLM
            context_chunks = [r.chunk.text for r in search_results]

            # 4. Generate answer
            answer = self.llm.generate(question, context_chunks)

            # 5. Build citations
            citations = [
                Citation(
                    rank=r.rank,
                    document=r.chunk.source_file,
                    page_number=r.chunk.page_number,
                    relevance_score=r.score,
                    excerpt=r.chunk.text,
                )
                for r in search_results
            ]

            return RAGResponse(
                question=question,
                answer=answer,
                citations=citations,
                retrieved_chunks=len(search_results),
                status="success",
            )

        except Exception as e:
            logger.error(f"Query error: {e}")
            return RAGResponse(
                question=question,
                answer=f"An error occurred while processing your query: {e}",
                status="error",
                error_message=str(e),
            )

    # ── Status ─────────────────────────────────

    def get_index_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()

"""
vector_store.py
---------------
FR3 – Embedding Generation : converts text chunks into dense vectors using
                              a local Sentence-Transformers model (no API key).
FR4 – Vector Database Storage: persists embeddings in a FAISS index on disk.
FR6 – Context Retrieval    : semantic similarity search over the index.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# LangChain + FAISS imports are deferred inside functions so that the module
# can be imported even if the heavy deps aren't installed yet (e.g. during
# unit-test collection).


# ── Embedding wrapper ─────────────────────────────────────────────────────────

class LocalEmbeddings:
    """
    Thin wrapper around sentence-transformers that satisfies
    LangChain's Embeddings interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    # LangChain Embeddings protocol
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# ── FAISS Vector Store ────────────────────────────────────────────────────────

INDEX_FILE    = "faiss_index"        # directory saved by FAISS
METADATA_FILE = "doc_metadata.pkl"  # maps FAISS int IDs → DocumentChunk metadata


class ComplianceVectorStore:
    """
    Manages a FAISS-backed vector store for banking policy documents.

    Usage:
        store = ComplianceVectorStore(persist_dir, embedding_model_name)
        store.add_documents(chunks)          # build / update index
        results = store.similarity_search(query, k=4)
    """

    def __init__(self, persist_dir: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir    = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings     = LocalEmbeddings(embedding_model)
        self._vectorstore   = None   # lazy: loaded/built on demand

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def index_path(self) -> Path:
        return self.persist_dir / INDEX_FILE

    @property
    def metadata_path(self) -> Path:
        return self.persist_dir / METADATA_FILE

    def _load_existing(self) -> bool:
        """Try to load a persisted index. Returns True if successful."""
        if self.index_path.exists():
            try:
                from langchain_community.vectorstores import FAISS
                self._vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s", self.index_path)
                return True
            except Exception as exc:
                logger.warning("Could not load existing index (%s) – will rebuild.", exc)
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, chunks) -> None:
        """
        Embed a list of DocumentChunk objects and upsert them into FAISS.
        Existing index is loaded first so that you can incrementally add docs.

        Args:
            chunks: List[DocumentChunk] from document_processor.py
        """
        from langchain_community.vectorstores import FAISS

        if not chunks:
            logger.warning("add_documents called with empty chunk list – skipping.")
            return

        # Convert to LangChain Documents
        lc_docs = [c.to_langchain_doc() for c in chunks]
        texts    = [d.page_content for d in lc_docs]
        metas    = [d.metadata     for d in lc_docs]

        logger.info("Embedding %d chunks …", len(chunks))

        if self._vectorstore is None and self._load_existing():
            # Append to existing
            self._vectorstore.add_texts(texts, metadatas=metas)
        else:
            # Build fresh
            self._vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metas)

        self._persist()
        logger.info("Vector store updated – %d total chunks.", self._vectorstore.index.ntotal)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[object, float]]:
        """
        Semantic similarity search.

        Returns:
            List of (LangChain Document, score) tuples, highest-score first.
            Score is cosine similarity (higher = more relevant).
        """
        if self._vectorstore is None:
            if not self._load_existing():
                raise RuntimeError(
                    "Vector store is empty. Please upload and process documents first."
                )
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        # FAISS returns L2 distance (lower = better); convert to similarity
        # score = 1 / (1 + distance)  so that higher is always better
        converted = [(doc, 1.0 / (1.0 + dist)) for doc, dist in results]
        return sorted(converted, key=lambda x: x[1], reverse=True)

    def get_retriever(self, k: int = 4):
        """Return a LangChain-compatible retriever object."""
        if self._vectorstore is None:
            self._load_existing()
        if self._vectorstore is None:
            raise RuntimeError("Vector store is empty.")
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    def document_count(self) -> int:
        """Return total number of chunks in the index."""
        if self._vectorstore is None:
            self._load_existing()
        if self._vectorstore is None:
            return 0
        return self._vectorstore.index.ntotal

    def _persist(self) -> None:
        """Save the FAISS index to disk."""
        self._vectorstore.save_local(str(self.index_path))
        logger.info("FAISS index persisted → %s", self.index_path)

    def reset(self) -> None:
        """Delete the persisted index (useful for re-indexing from scratch)."""
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        self._vectorstore = None
        logger.info("Vector store reset.")

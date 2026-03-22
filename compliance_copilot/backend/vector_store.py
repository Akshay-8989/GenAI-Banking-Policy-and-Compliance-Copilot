"""
vector_store.py
---------------
FAISS-based vector database for storing and retrieving document embeddings.
Covers BRD FR4 – Vector Database Storage and FR6 – Context Retrieval.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from loguru import logger

from document_processor import DocumentChunk


# ──────────────────────────────────────────────
# Search Result Model
# ──────────────────────────────────────────────

class SearchResult:
    """Holds a retrieved chunk with its relevance score."""

    def __init__(self, chunk: DocumentChunk, score: float, rank: int):
        self.chunk = chunk
        self.score = score          # higher = more similar (cosine)
        self.rank = rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "text": self.chunk.text,
            "source_file": self.chunk.source_file,
            "page_number": self.chunk.page_number,
            "chunk_id": self.chunk.chunk_id,
        }

    def __repr__(self):
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, source={self.chunk.source_file}:p{self.chunk.page_number})"


# ──────────────────────────────────────────────
# FAISS Vector Store
# ──────────────────────────────────────────────

class FAISSVectorStore:
    """
    Persistent FAISS index with full chunk metadata.
    Stores:
      - FAISS flat L2 index (inner-product for normalised embeddings = cosine)
      - Parallel list of DocumentChunk objects (pickle)
      - Index metadata JSON
    """

    INDEX_FILE = "faiss.index"
    CHUNKS_FILE = "chunks.pkl"
    META_FILE = "metadata.json"

    def __init__(self, store_path: str, embedding_dim: int = 384):
        self.store_path = Path(store_path)
        self.embedding_dim = embedding_dim
        self._index = None
        self._chunks: List[DocumentChunk] = []
        self._meta: Dict[str, Any] = {"total_chunks": 0, "documents": []}

    # ── Persistence ────────────────────────────

    def _ensure_dir(self):
        self.store_path.mkdir(parents=True, exist_ok=True)

    def save(self):
        """Persist index + chunks + metadata to disk."""
        import faiss
        self._ensure_dir()

        # FAISS index
        faiss.write_index(self._index, str(self.store_path / self.INDEX_FILE))

        # Chunk metadata
        with open(self.store_path / self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self._chunks, f)

        # Human-readable metadata
        self._meta["total_chunks"] = len(self._chunks)
        with open(self.store_path / self.META_FILE, "w") as f:
            json.dump(self._meta, f, indent=2)

        logger.success(f"Vector store saved → {self.store_path}  ({len(self._chunks)} chunks)")

    def load(self) -> bool:
        """Load persisted index. Returns True if loaded successfully."""
        import faiss

        index_path = self.store_path / self.INDEX_FILE
        chunks_path = self.store_path / self.CHUNKS_FILE

        if not index_path.exists() or not chunks_path.exists():
            logger.info("No existing vector store found. A fresh index will be created.")
            return False

        self._index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        if (self.store_path / self.META_FILE).exists():
            with open(self.store_path / self.META_FILE) as f:
                self._meta = json.load(f)

        logger.success(f"Vector store loaded ← {self.store_path}  ({len(self._chunks)} chunks)")
        self.embedding_dim = self._index.d
        return True

    def exists(self) -> bool:
        return (self.store_path / self.INDEX_FILE).exists()

    # ── Indexing ───────────────────────────────

    def _create_index(self):
        """Create a new flat inner-product FAISS index."""
        import faiss
        # IndexFlatIP gives cosine similarity when vectors are L2-normalised
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Created new FAISS IndexFlatIP (dim={self.embedding_dim})")

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
        document_name: Optional[str] = None,
    ):
        """
        Add chunks + their embeddings to the index.
        embeddings: float32 numpy array (N, dim), already L2-normalised.
        """
        import faiss

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Chunk count ({len(chunks)}) ≠ embedding count ({embeddings.shape[0]})")

        if self._index is None:
            self.embedding_dim = embeddings.shape[1]
            self._create_index()

        # Ensure float32 and L2-normalised (sentence-transformers returns normalised)
        vecs = embeddings.astype(np.float32)
        faiss.normalize_L2(vecs)  # idempotent if already normalised

        self._index.add(vecs)
        self._chunks.extend(chunks)

        if document_name and document_name not in self._meta.get("documents", []):
            self._meta.setdefault("documents", []).append(document_name)

        logger.info(f"Added {len(chunks)} chunks. Total in index: {len(self._chunks)}")

    # ── Retrieval ──────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for top-k most relevant chunks.
        query_embedding: shape (1, dim) or (dim,), float32.
        Returns SearchResult list ordered by descending relevance.
        """
        import faiss

        if self._index is None or self._index.ntotal == 0:
            logger.warning("Vector store is empty. Index documents first.")
            return []

        # Reshape and normalise query
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:                          # FAISS returns -1 for padding
                continue
            if score < score_threshold:
                continue
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=float(score),
                rank=rank,
            ))

        logger.debug(f"Search returned {len(results)} results (top score: {results[0].score:.4f if results else 'n/a'})")
        return results

    # ── Utilities ──────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self._chunks),
            "total_vectors": self._index.ntotal if self._index else 0,
            "embedding_dim": self.embedding_dim,
            "documents": self._meta.get("documents", []),
            "store_path": str(self.store_path),
        }

    def clear(self):
        """Wipe the index completely."""
        import faiss
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._chunks = []
        self._meta = {"total_chunks": 0, "documents": []}
        logger.warning("Vector store cleared.")

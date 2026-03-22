"""
embeddings.py
-------------
Converts document chunks into dense vector embeddings.
Uses sentence-transformers (all-MiniLM-L6-v2) – runs fully offline.
Covers BRD FR3 – Embedding Generation.
"""

import os
from typing import List, Optional

import numpy as np
from loguru import logger

from document_processor import DocumentChunk


# ──────────────────────────────────────────────
# Embedding Model Wrapper
# ──────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps sentence-transformers for chunk-level embedding generation.
    Model is loaded once and cached.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        logger.info(f"EmbeddingModel initialised (model will load on first use): {model_name}")

    def _load(self):
        """Lazy load to avoid startup cost if embeddings already cached."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.success(f"Embedding model loaded. Dimension: {self.get_dimension()}")

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of strings.
        Returns float32 numpy array of shape (N, dim).
        """
        self._load()
        if not texts:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})…")
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )
        logger.success(f"Generated embeddings: shape {vectors.shape}")
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed_texts([query], show_progress=False)

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> np.ndarray:
        """Convenience method: embed a list of DocumentChunk objects."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)

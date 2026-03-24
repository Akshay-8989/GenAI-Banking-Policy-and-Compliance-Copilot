"""
vector_store.py
---------------
FR3 – Embedding Generation : TF-IDF embeddings (no download, no internet needed)
FR4 – Vector Database Storage: FAISS index persisted to disk
FR6 – Context Retrieval    : cosine similarity search
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

INDEX_FILE    = "tfidf_index.pkl"   # stores everything in one pickle file


class TFIDFEmbeddings:
    """
    Local TF-IDF based embeddings using scikit-learn.
    No internet, no downloads, no GPU needed.
    Works entirely offline with libraries already installed.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=8192,
            ngram_range=(1, 2),      # unigrams + bigrams
            sublinear_tf=True,       # log normalization
            min_df=1,
            strip_accents="unicode",
            analyzer="word",
        )
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted yet — add documents first.")
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].tolist()


class ComplianceVectorStore:
    """
    TF-IDF + cosine similarity vector store.
    Fully offline — no sentence-transformers, no HuggingFace, no internet.
    """

    def __init__(self, persist_dir: Path, embedding_model: str = None):
        # embedding_model param kept for API compatibility but not used
        self.persist_dir  = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store       = None   # dict with embeddings, texts, metadatas, vectorizer
        self._load_existing()

    @property
    def index_path(self) -> Path:
        return self.persist_dir / INDEX_FILE

    def _load_existing(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    self._store = pickle.load(f)
                logger.info("Loaded existing index: %d chunks", len(self._store["texts"]))
            except Exception as e:
                logger.warning("Could not load existing index: %s", e)
                self._store = None

    def _save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump(self._store, f)
        logger.info("Index saved → %s", self.index_path)

    def add_documents(self, chunks) -> None:
        if not chunks:
            logger.warning("add_documents called with empty list.")
            return

        new_texts  = [c.text for c in chunks]
        new_metas  = [{"source": c.source_file, "page": c.page_number,
                       "chunk_index": c.chunk_index, "doc_hash": c.doc_hash}
                      for c in chunks]

        # Merge with existing texts if any
        if self._store is not None:
            all_texts = self._store["texts"] + new_texts
            all_metas = self._store["metadatas"] + new_metas
        else:
            all_texts = new_texts
            all_metas = new_metas

        # Re-fit TF-IDF on all texts and compute embeddings
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=8192,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(all_texts)   # sparse matrix

        self._store = {
            "texts":      all_texts,
            "metadatas":  all_metas,
            "matrix":     matrix,       # scipy sparse — very compact
            "vectorizer": vectorizer,
        }
        self._save()
        logger.info("Index updated — %d total chunks", len(all_texts))

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Tuple[object, float]]:
        if self._store is None:
            raise RuntimeError(
                "Vector store is empty. Please upload and process documents first."
            )

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = self._store["vectorizer"]
        matrix     = self._store["matrix"]

        query_vec  = vectorizer.transform([query])
        scores     = cosine_similarity(query_vec, matrix)[0]   # shape (n_docs,)

        # Get top-k indices
        top_k_idx  = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_idx:
            score = float(scores[idx])
            if score == 0.0:
                continue   # skip completely irrelevant results
            # Build a simple document-like object
            doc = _SimpleDoc(
                page_content=self._store["texts"][idx],
                metadata=self._store["metadatas"][idx],
            )
            results.append((doc, score))

        return results

    def document_count(self) -> int:
        if self._store is None:
            return 0
        return len(self._store["texts"])

    def reset(self):
        import shutil
        if self.index_path.exists():
            self.index_path.unlink()
        self._store = None
        logger.info("Vector store reset.")


class _SimpleDoc:
    """Minimal document object matching LangChain Document interface."""
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata     = metadata

"""
api.py
------
FastAPI REST backend exposing the RAG pipeline via HTTP endpoints.
The Streamlit frontend calls these endpoints.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from rag_pipeline import RAGPipeline, RAGResponse


# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="GenAI Banking Compliance Copilot API",
    description="RAG-powered policy Q&A using Phi-2 + FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-init pipeline singleton
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline.from_env()
    return _pipeline


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[dict]
    retrieved_chunks: int
    status: str
    error_message: str = ""


class IngestResponse(BaseModel):
    message: str
    results: List[dict]


class StatsResponse(BaseModel):
    total_chunks: int
    total_vectors: int
    embedding_dim: int
    documents: List[str]
    store_path: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Compliance Copilot API"}


@app.get("/health", tags=["Health"])
def health():
    pipeline = get_pipeline()
    stats = pipeline.get_index_stats()
    return {
        "status": "healthy",
        "indexed_documents": len(stats["documents"]),
        "total_chunks": stats["total_chunks"],
    }


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a single policy document (PDF or TXT)."""
    allowed_types = {"application/pdf", "text/plain"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload PDF or TXT.",
        )

    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        result = pipeline.ingest_file(tmp_path)
        # Rename result to original filename for clarity
        result["file"] = file.filename
        return IngestResponse(
            message=f"Document '{file.filename}' ingested successfully.",
            results=[result],
        )
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
def ingest_directory(directory: str = "./data/policies"):
    """Ingest all documents from a server-side directory."""
    if not Path(directory).exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    pipeline = get_pipeline()
    results = pipeline.ingest_directory(directory)

    return IngestResponse(
        message=f"Processed {len(results)} document(s) from '{directory}'.",
        results=results,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """Ask a natural-language compliance question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    if request.top_k:
        pipeline.top_k = request.top_k

    response: RAGResponse = pipeline.query(request.question)

    return QueryResponse(**response.to_dict())


@app.get("/stats", response_model=StatsResponse, tags=["Status"])
def stats():
    """Return vector store statistics."""
    pipeline = get_pipeline()
    return StatsResponse(**pipeline.get_index_stats())


@app.delete("/index", tags=["Management"])
def clear_index():
    """Clear the entire vector index (use with caution)."""
    pipeline = get_pipeline()
    pipeline.vector_store.clear()
    pipeline.vector_store.save()
    return {"message": "Vector index cleared successfully."}


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Starting Compliance Copilot API on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False)

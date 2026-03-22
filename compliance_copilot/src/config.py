"""
config.py
---------
Centralised configuration for the Compliance Copilot.
All tunable parameters live here; override via environment variables or .env file.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
UPLOAD_DIR      = Path(os.getenv("UPLOAD_PATH",      str(BASE_DIR / "data" / "uploads")))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_PATH", str(BASE_DIR / "data" / "vectorstore")))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
PHI2_MODEL_NAME = os.getenv("PHI2_MODEL_NAME", "microsoft/phi-2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE    = float(os.getenv("TEMPERATURE",  "0.1"))   # low → factual answers

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "Banking Policy & Compliance Copilot")

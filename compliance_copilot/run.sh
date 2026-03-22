#!/bin/bash
# ============================================================
# run.sh – Start the Compliance Copilot (backend + frontend)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example"
fi

source .env 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      GenAI Banking Compliance Copilot v1.0          ║"
echo "║      Phi-2 SLM  ·  FAISS  ·  Streamlit             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check Python ─────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.9+ first."
    exit 1
fi

echo "[INFO] Python: $(python3 --version)"

# ── Install dependencies ──────────────────────────────────
echo ""
echo "[STEP 1] Installing dependencies…"
pip install -r requirements.txt -q

# ── Pre-ingest sample documents ──────────────────────────
echo ""
echo "[STEP 2] Ingesting sample policy documents…"
python3 ingest.py --dir ./data/policies

# ── Start FastAPI backend ─────────────────────────────────
echo ""
echo "[STEP 3] Starting FastAPI backend on port ${API_PORT:-8000}…"
cd backend
uvicorn api:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --log-level warning &
BACKEND_PID=$!
cd ..

# Wait for API to come up
sleep 3
if curl -s "http://localhost:${API_PORT:-8000}/health" > /dev/null 2>&1; then
    echo "[OK]  Backend running → http://localhost:${API_PORT:-8000}"
    echo "[OK]  API Docs        → http://localhost:${API_PORT:-8000}/docs"
else
    echo "[WARN] Backend not responding yet (may still be loading Phi-2)"
fi

# ── Start Streamlit frontend ──────────────────────────────
echo ""
echo "[STEP 4] Starting Streamlit UI on port ${STREAMLIT_PORT:-8501}…"
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │  Open → http://localhost:${STREAMLIT_PORT:-8501}            │"
echo "  └─────────────────────────────────────────────┘"
echo ""

# Set API URL for frontend
export API_BASE_URL="http://localhost:${API_PORT:-8000}"

cd frontend
streamlit run app.py \
    --server.port "${STREAMLIT_PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --theme.base dark \
    --theme.primaryColor "#00a8e8" \
    --theme.backgroundColor "#0a0e17" \
    --theme.secondaryBackgroundColor "#0d1120"

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT

"""
app.py  –  Streamlit Frontend
------------------------------
Banking Compliance Copilot chat interface.
Design: Dark financial terminal aesthetic – authoritative, precise, trustworthy.
"""

import os
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
BANK_NAME = os.getenv("BANK_NAME", "SecureBank")
APP_TITLE = os.getenv("APP_TITLE", "Compliance Copilot")

EXAMPLE_QUERIES = [
    "What documents are required for KYC verification?",
    "What is the AML suspicious transaction reporting threshold?",
    "What are the penalties for non-compliance with AML regulations?",
    "Describe the customer due diligence (CDD) process.",
    "What is the policy for high-risk customer accounts?",
    "How long must KYC records be retained?",
]

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title=f"{BANK_NAME} · {APP_TITLE}",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  .stApp {
    background: #0a0e17;
    color: #c8d6e5;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2d45;
  }

  [data-testid="stSidebar"] .stMarkdown {
    color: #8899aa;
  }

  /* ── Header strip ── */
  .copilot-header {
    background: linear-gradient(135deg, #0d1829 0%, #0a1628 50%, #061020 100%);
    border: 1px solid #1a3050;
    border-radius: 4px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    overflow: hidden;
  }

  .copilot-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0047ab, #00a8e8, #00e5a0);
  }

  .copilot-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #e0eeff;
    margin: 0;
    letter-spacing: 0.05em;
  }

  .copilot-header p {
    font-size: 0.8rem;
    color: #5577aa;
    margin: 2px 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
  }

  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00e5a0;
    box-shadow: 0 0 8px #00e5a0;
    flex-shrink: 0;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Chat messages ── */
  .msg-user {
    background: #0e1f35;
    border: 1px solid #1a3a5c;
    border-radius: 4px 4px 4px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #c8d6e5;
    position: relative;
  }

  .msg-user::before {
    content: 'YOU';
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3a6aaa;
    position: absolute;
    top: -10px; left: 0;
    letter-spacing: 0.12em;
  }

  .msg-bot {
    background: #06111e;
    border: 1px solid #0d2a40;
    border-left: 3px solid #00a8e8;
    border-radius: 0 4px 4px 4px;
    padding: 14px 18px;
    margin: 8px 0 20px 0;
    font-size: 0.9rem;
    color: #c8d6e5;
    position: relative;
    line-height: 1.65;
  }

  .msg-bot::before {
    content: 'COPILOT';
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #00a8e8;
    position: absolute;
    top: -10px; left: 0;
    letter-spacing: 0.12em;
  }

  .msg-bot.no-context {
    border-left-color: #e8a000;
  }

  .msg-bot.error {
    border-left-color: #e84040;
  }

  /* ── Citations ── */
  .citation-box {
    background: #071520;
    border: 1px solid #0d2535;
    border-radius: 3px;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  .citation-header {
    color: #3a6aaa;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
  }

  .citation-item {
    display: flex;
    gap: 10px;
    align-items: baseline;
    margin: 4px 0;
    color: #5577aa;
    border-bottom: 1px solid #0d1e2e;
    padding-bottom: 4px;
  }

  .citation-rank {
    color: #00a8e8;
    font-weight: 600;
    min-width: 20px;
  }

  .citation-score {
    color: #00e5a0;
    min-width: 55px;
  }

  .citation-source {
    color: #7a99bb;
  }

  /* ── Input area ── */
  .stTextInput input, .stTextArea textarea {
    background: #0a1628 !important;
    border: 1px solid #1a3050 !important;
    border-radius: 3px !important;
    color: #c8d6e5 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
  }

  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00a8e8 !important;
    box-shadow: 0 0 0 1px #00a8e822 !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #0047ab !important;
    border: 1px solid #0060d0 !important;
    color: #e0eeff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 3px !important;
    transition: all 0.15s !important;
  }

  .stButton > button:hover {
    background: #0060d0 !important;
    border-color: #00a8e8 !important;
    box-shadow: 0 0 12px #0047ab44 !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #0a1628;
    border: 1px dashed #1a3050;
    border-radius: 4px;
    padding: 8px;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background: #0a1628;
    border: 1px solid #1a3050;
    border-radius: 4px;
    padding: 12px;
  }

  [data-testid="stMetricLabel"] { color: #5577aa !important; font-size: 0.75rem !important; }
  [data-testid="stMetricValue"] { color: #00e5a0 !important; font-family: 'IBM Plex Mono', monospace !important; }

  /* ── Divider ── */
  hr { border-color: #1a2a3a !important; }

  /* ── Expander ── */
  details { border: 1px solid #1a2a3a !important; border-radius: 3px !important; }
  summary { color: #5577aa !important; font-size: 0.8rem !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0e17; }
  ::-webkit-scrollbar-thumb { background: #1a3050; border-radius: 2px; }

  /* ── Tag pills ── */
  .tag {
    display: inline-block;
    background: #071828;
    border: 1px solid #0d2a40;
    color: #3a6aaa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin: 2px;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],          # list of {role, content, citations, status}
        "api_connected": False,
        "indexed_docs": [],
        "total_chunks": 0,
        "top_k": 4,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ──────────────────────────────────────────────
# API Helper Functions
# ──────────────────────────────────────────────

def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_ingest_file(uploaded_file) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/ingest/file",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
            timeout=120,
        )
        return r.json()
    except Exception as e:
        return {"message": f"Error: {e}", "results": []}


def api_query(question: str, top_k: int = 4) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        return r.json()
    except Exception as e:
        return {
            "question": question,
            "answer": f"API connection error: {e}. Make sure the backend is running.",
            "citations": [],
            "retrieved_chunks": 0,
            "status": "error",
        }


def api_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json()
    except Exception:
        return {}


def refresh_stats():
    stats = api_stats()
    if stats:
        st.session_state.indexed_docs = stats.get("documents", [])
        st.session_state.total_chunks = stats.get("total_chunks", 0)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 16px 0">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;color:#e0eeff;font-weight:500">
        🏦 {BANK_NAME}
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#3a6aaa;letter-spacing:0.1em">
        COMPLIANCE INTELLIGENCE SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Connection check
    health = api_health()
    if health:
        st.session_state.api_connected = True
        st.success("⬤  API Connected", icon=None)
    else:
        st.session_state.api_connected = False
        st.error("⬤  API Offline – start backend first")

    st.markdown("---")

    # ── Document Upload ──────────────────
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#3a6aaa;letter-spacing:0.12em;margin-bottom:8px">DOCUMENT INGESTION</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Policy Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("▶  INGEST DOCUMENTS", use_container_width=True):
        progress = st.progress(0)
        for i, f in enumerate(uploaded):
            with st.spinner(f"Ingesting {f.name}…"):
                result = api_ingest_file(f)
            progress.progress((i + 1) / len(uploaded))
            if "results" in result and result["results"]:
                r = result["results"][0]
                if r.get("status") == "success":
                    st.success(f"✓ {f.name} → {r.get('chunks',0)} chunks")
                else:
                    st.error(f"✗ {f.name}: {r.get('error','failed')}")
        refresh_stats()
        progress.empty()

    st.markdown("---")

    # ── Index Stats ──────────────────────
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#3a6aaa;letter-spacing:0.12em;margin-bottom:8px">INDEX STATUS</div>', unsafe_allow_html=True)

    if st.button("↺  REFRESH STATS", use_container_width=True):
        refresh_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", len(st.session_state.indexed_docs))
    with col2:
        st.metric("Chunks", st.session_state.total_chunks)

    if st.session_state.indexed_docs:
        with st.expander("Indexed files"):
            for d in st.session_state.indexed_docs:
                st.markdown(f'<span class="tag">📄 {d}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Settings ─────────────────────────
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#3a6aaa;letter-spacing:0.12em;margin-bottom:8px">RETRIEVAL SETTINGS</div>', unsafe_allow_html=True)
    st.session_state.top_k = st.slider("Top-K chunks", 1, 8, 4)

    st.markdown("---")

    # ── Clear chat ───────────────────────
    if st.button("🗑  CLEAR CHAT", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#1e3050;margin-top:20px;text-align:center">Phi-2 · FAISS · Sentence Transformers<br>RAG v1.0</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Panel
# ──────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="copilot-header">
  <div class="status-dot"></div>
  <div>
    <h1>⚖ {APP_TITLE}</h1>
    <p>Retrieval-Augmented Generation · Phi-2 SLM · FAISS Vector Search</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Quick-start example queries
if not st.session_state.messages:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#3a6aaa;letter-spacing:0.1em;margin-bottom:12px">EXAMPLE QUERIES</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES):
        with cols[i % 2]:
            if st.button(q, key=f"eq_{i}", use_container_width=True):
                st.session_state._pending_query = q
                st.rerun()

# ── Render chat history ──────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            status_class = msg.get("status", "success")
            st.markdown(f'<div class="msg-bot {status_class}">{msg["content"]}</div>', unsafe_allow_html=True)

            # Citations
            citations = msg.get("citations", [])
            if citations:
                citation_rows = ""
                for c in citations:
                    doc = c.get("document", "unknown")
                    page = c.get("page_number", "?")
                    score = c.get("relevance_score", 0)
                    citation_rows += f"""
                    <div class="citation-item">
                      <span class="citation-rank">#{c.get('rank','-')}</span>
                      <span class="citation-score">▲ {score:.3f}</span>
                      <span class="citation-source">📄 {doc}  ·  p.{page}</span>
                    </div>"""
                st.markdown(f"""
                <div class="citation-box">
                  <div class="citation-header">SOURCE CITATIONS  ({len(citations)} retrieved)</div>
                  {citation_rows}
                </div>""", unsafe_allow_html=True)

                # Expandable excerpts
                with st.expander("View retrieved context"):
                    for c in citations:
                        st.markdown(f"**[{c.get('rank')}] {c.get('document')} – Page {c.get('page_number')}** (score: {c.get('relevance_score',0):.4f})")
                        st.caption(c.get("excerpt", "")[:500])
                        st.markdown("---")

# ── Input bar ───────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Handle pending query from example buttons
if hasattr(st.session_state, "_pending_query"):
    pending = st.session_state._pending_query
    del st.session_state._pending_query
    user_input = pending
else:
    user_input = None

col_input, col_btn = st.columns([6, 1])
with col_input:
    typed = st.text_input(
        "Ask a compliance question…",
        value="",
        key="query_input",
        placeholder="e.g. What are the KYC document requirements for individual customers?",
        label_visibility="collapsed",
    )
with col_btn:
    send_clicked = st.button("SEND ▶", use_container_width=True)

if user_input is None and (send_clicked or typed) and typed:
    user_input = typed

# ── Process query ────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Searching policy documents…"):
        response = api_query(user_input, top_k=st.session_state.top_k)

    st.session_state.messages.append({
        "role": "bot",
        "content": response.get("answer", "No answer returned."),
        "citations": response.get("citations", []),
        "status": response.get("status", "success"),
    })

    st.rerun()

# ── Empty state ──────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:48px 0;color:#1e3a50">
      <div style="font-size:3rem;margin-bottom:12px">⚖</div>
      <div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;letter-spacing:0.1em">
        INGEST POLICY DOCUMENTS · ASK COMPLIANCE QUESTIONS
      </div>
    </div>
    """, unsafe_allow_html=True)

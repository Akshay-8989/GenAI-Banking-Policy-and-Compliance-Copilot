"""
app.py
------
Streamlit UI for the GenAI Banking Policy & Compliance Copilot.

Run with:
    streamlit run app.py

Features:
  • Sidebar: upload PDFs, indexing status, document list
  • Main area: chat interface with Q&A history
  • Expandable source citations for every answer
  • Session-persistent chat history
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import streamlit as st

# ── Add project root to sys.path so `src` is importable ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    APP_TITLE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_NEW_TOKENS,
    PHI2_MODEL_NAME,
    TEMPERATURE,
    TOP_K_RESULTS,
    UPLOAD_DIR,
    VECTORSTORE_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a2744 0%, #0d3b6e 50%, #1565c0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }

    /* Chat bubbles */
    .user-bubble {
        background: #e3f2fd;
        border-left: 4px solid #1565c0;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .bot-bubble {
        background: #f8f9fa;
        border-left: 4px solid #2e7d32;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .source-box {
        background: #fffde7;
        border: 1px solid #f9a825;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f5f7fa;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Status badges */
    .badge-green { color: #2e7d32; font-weight: 600; }
    .badge-orange { color: #e65100; font-weight: 600; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def _init_session():
    defaults = {
        "chat_history":     [],    # list of {"role": "user"|"assistant", "content": ..., "sources": [...]}
        "pipeline":         None,
        "indexed_files":    [],    # filenames that have been indexed
        "model_loaded":     False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ── Pipeline singleton ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline …")
def get_pipeline():
    from src.rag_pipeline import ComplianceRAGPipeline
    return ComplianceRAGPipeline(
        top_k          = TOP_K_RESULTS,
        model_name     = PHI2_MODEL_NAME,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature    = TEMPERATURE,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Compliance Copilot")
    st.markdown("---")

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload KYC, AML, Loan, or Risk Management policy PDFs",
    )

    if uploaded_files:
        pipeline = get_pipeline()
        for uf in uploaded_files:
            if uf.name not in st.session_state["indexed_files"]:
                dest = UPLOAD_DIR / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                with st.spinner(f"Indexing {uf.name} …"):
                    try:
                        from src.document_processor import load_and_chunk_pdf
                        chunks = load_and_chunk_pdf(dest, CHUNK_SIZE, CHUNK_OVERLAP)
                        pipeline.vectorstore.add_documents(chunks)
                        st.session_state["indexed_files"].append(uf.name)
                        st.success(f"✅ Indexed: {uf.name} ({len(chunks)} chunks)")
                    except Exception as exc:
                        st.error(f"❌ Failed: {uf.name}\n{exc}")

    st.markdown("---")

    # ── Index status ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Index Status")
    try:
        pipeline  = get_pipeline()
        doc_count = pipeline.vectorstore.document_count()
        indexed   = st.session_state["indexed_files"]

        if doc_count > 0:
            st.markdown(f'<span class="badge-green">● Online</span>  —  {doc_count} chunks indexed', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-orange">● No documents indexed yet</span>', unsafe_allow_html=True)

        if indexed:
            st.markdown("**Indexed files:**")
            for fname in indexed:
                st.markdown(f"  • {fname}")
    except Exception:
        st.info("Pipeline not loaded yet.")

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Settings"):
        st.markdown(f"**Embedding model:** `{EMBEDDING_MODEL}`")
        st.markdown(f"**LLM:** `{PHI2_MODEL_NAME}`")
        st.markdown(f"**Chunk size:** {CHUNK_SIZE} chars")
        st.markdown(f"**Top-K results:** {TOP_K_RESULTS}")
        st.markdown(f"**Max new tokens:** {MAX_NEW_TOKENS}")
        st.markdown(f"**Temperature:** {TEMPERATURE}")

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()


# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🏦 {APP_TITLE}</h1>
    <p>Ask questions about KYC, AML, loan policies, and risk management guidelines</p>
</div>
""", unsafe_allow_html=True)

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    "What documents are required for KYC verification?",
    "What is the AML reporting threshold?",
    "What are the risk assessment guidelines for loan approval?",
    "What happens if a customer fails the KYC check?",
    "Describe the compliance escalation process.",
]

with st.expander("💡 Example Questions", expanded=False):
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES):
        col = cols[i % 2]
        if col.button(q, key=f"ex_{i}"):
            st.session_state["pending_query"] = q

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 <strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">🤖 <strong>Copilot:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Source citations
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} documents cited)", expanded=False):
                    for s in sources:
                        st.markdown(
                            f'<div class="source-box">'
                            f'📄 <strong>{s["document_name"]}</strong> — Page {s["page_number"]} '
                            f'| Relevance: {s["relevance_score"]:.0%}<br>'
                            f'<em>…{s["excerpt"]}…</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ── Input box ─────────────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_btn = st.columns([5, 1])

with col_input:
    # Pre-fill if an example query was clicked
    default_val = st.session_state.pop("pending_query", "")
    user_input  = st.text_input(
        "Ask a compliance question …",
        value=default_val,
        placeholder="e.g. What documents are required for KYC?",
        label_visibility="collapsed",
        key="user_input_box",
    )

with col_btn:
    submit = st.button("Send 🚀", use_container_width=True)

# ── Handle query ──────────────────────────────────────────────────────────────
if submit and user_input.strip():
    question = user_input.strip()

    # Add user message
    st.session_state["chat_history"].append({
        "role":    "user",
        "content": question,
    })

    # Run RAG pipeline
    with st.spinner("🔍 Searching policy documents and generating answer …"):
        t0       = time.time()
        pipeline = get_pipeline()
        response = pipeline.query(question)
        elapsed  = time.time() - t0

    # Add assistant message
    sources_serializable = [
        {
            "document_name":   s.document_name,
            "page_number":     s.page_number,
            "excerpt":         s.excerpt,
            "relevance_score": s.relevance_score,
        }
        for s in response.sources
    ]
    st.session_state["chat_history"].append({
        "role":    "assistant",
        "content": f"{response.answer}\n\n⏱ *Generated in {elapsed:.1f}s*",
        "sources": sources_serializable,
    })

    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state["chat_history"]:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #9e9e9e;">
        <h3>👋 Welcome to the Compliance Copilot</h3>
        <p>Upload policy PDF documents using the sidebar, then ask any compliance question.</p>
        <p>The AI will answer using only your uploaded documents and cite the exact sources.</p>
    </div>
    """, unsafe_allow_html=True)

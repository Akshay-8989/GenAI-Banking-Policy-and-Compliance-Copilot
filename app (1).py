"""
app.py  –  GenAI Banking Policy & Compliance Copilot
Run with:  streamlit run app.py
"""
from __future__ import annotations
import logging, sys, time, traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    APP_TITLE, CHUNK_OVERLAP, CHUNK_SIZE,
    MAX_NEW_TOKENS, PHI2_MODEL_NAME, TEMPERATURE, TOP_K_RESULTS,
    UPLOAD_DIR,
)

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a2744 0%,#0d3b6e 50%,#1565c0 100%);
    padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;color:white;}
.main-header h1{margin:0;font-size:1.8rem;}
.main-header p{margin:0.3rem 0 0;opacity:0.8;font-size:0.9rem;}
.user-bubble{background:#e3f2fd;border-left:4px solid #1565c0;
    padding:0.8rem 1rem;border-radius:8px;margin:0.5rem 0;}
.bot-bubble{background:#f8f9fa;border-left:4px solid #2e7d32;
    padding:0.8rem 1rem;border-radius:8px;margin:0.5rem 0;}
.source-box{background:#fffde7;border:1px solid #f9a825;border-radius:6px;
    padding:0.6rem 1rem;margin:0.3rem 0;font-size:0.85rem;}
.err-box{background:#fff3f3;border-left:4px solid #c62828;
    padding:0.8rem 1rem;border-radius:8px;margin:0.5rem 0;font-size:0.85rem;}
.badge-green{color:#2e7d32;font-weight:600;}
.badge-orange{color:#e65100;font-weight:600;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"chat_history": [], "indexed_files": [], "input_counter": 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.rag_pipeline import ComplianceRAGPipeline
    return ComplianceRAGPipeline(
        top_k=TOP_K_RESULTS, model_name=PHI2_MODEL_NAME,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Compliance Copilot")
    st.markdown("---")
    st.markdown("### 📄 Upload Policy Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True,
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
                        st.error(f"❌ Failed: {uf.name}\n\n{traceback.format_exc()}")

    st.markdown("---")
    st.markdown("### 📊 Index Status")
    try:
        pipeline  = get_pipeline()
        doc_count = pipeline.vectorstore.document_count()
        indexed   = st.session_state["indexed_files"]
        if doc_count > 0:
            st.markdown(f'<span class="badge-green">● Online — {doc_count} chunks indexed</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-orange">● No documents indexed yet</span>',
                        unsafe_allow_html=True)
        for fname in indexed:
            st.markdown(f"  • {fname}")
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")

    st.markdown("---")
    with st.expander("⚙️ Settings"):
        st.markdown(f"**LLM path:** `{PHI2_MODEL_NAME}`")
        st.markdown(f"**Embedding:** `TF-IDF (offline)`")
        st.markdown(f"**Chunk size:** {CHUNK_SIZE} | **Top-K:** {TOP_K_RESULTS}")
        st.markdown(f"**Max tokens:** {MAX_NEW_TOKENS} | **Temp:** {TEMPERATURE}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["input_counter"] += 1
        st.rerun()

# ── Main panel ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🏦 {APP_TITLE}</h1>
    <p>Ask questions about KYC, AML, loan policies, and risk management guidelines</p>
</div>
""", unsafe_allow_html=True)

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLES = [
    "What documents are required for KYC verification?",
    "What is the AML reporting threshold?",
    "What are the risk assessment guidelines for loan approval?",
    "What happens if a customer fails the KYC check?",
    "Describe the compliance escalation process.",
]
with st.expander("💡 Example Questions", expanded=False):
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex_{i}"):
            st.session_state["pending_query"] = q
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">👤 <b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True)
    else:
        # Show error banner if there was an LLM error
        if msg.get("error"):
            st.markdown(
                f'<div class="err-box">⚠️ {msg["error"]}</div>',
                unsafe_allow_html=True)
        st.markdown(
            f'<div class="bot-bubble">🤖 <b>Copilot:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True)
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} cited)", expanded=False):
                for s in sources:
                    st.markdown(
                        f'<div class="source-box">'
                        f'📄 <b>{s["document_name"]}</b> — Page {s["page_number"]} '
                        f'| Relevance: {s["relevance_score"]:.0%}<br>'
                        f'<em>…{s["excerpt"]}…</em></div>',
                        unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.form(key=f"chat_form_{st.session_state['input_counter']}", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        default_val = st.session_state.pop("pending_query", "")
        user_input  = st.text_input(
            "Question", value=default_val,
            placeholder="e.g. What documents are required for KYC?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send 🚀", use_container_width=True)

# ── Process ───────────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state["chat_history"].append({"role": "user", "content": question})

    with st.spinner("🔍 Searching policy documents and generating answer…"):
        try:
            t0       = time.time()
            pipeline = get_pipeline()
            response = pipeline.query(question)
            elapsed  = time.time() - t0

            answer_text = response.answer
            if answer_text and answer_text.strip():
                answer_text += f"\n\n⏱ *Generated in {elapsed:.1f}s*"
            else:
                answer_text = "⚠️ The model returned an empty response. Please try rephrasing your question."

            st.session_state["chat_history"].append({
                "role":    "assistant",
                "content": answer_text,
                "error":   response.error,
                "sources": [
                    {
                        "document_name":   s.document_name,
                        "page_number":     s.page_number,
                        "excerpt":         s.excerpt,
                        "relevance_score": s.relevance_score,
                    }
                    for s in response.sources
                ],
            })
        except Exception as exc:
            # Catch anything unexpected and show it clearly
            full_tb = traceback.format_exc()
            st.session_state["chat_history"].append({
                "role":    "assistant",
                "content": f"⚠️ Unexpected error:\n```\n{full_tb}\n```",
                "error":   str(exc),
                "sources": [],
            })

    st.session_state["input_counter"] += 1
    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state["chat_history"]:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#9e9e9e;">
        <h3>👋 Welcome to the Compliance Copilot</h3>
        <p>Upload a policy PDF using the sidebar, then ask any compliance question.</p>
        <p>The AI will search the document and return cited answers.</p>
    </div>
    """, unsafe_allow_html=True)

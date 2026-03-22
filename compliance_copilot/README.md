# 🏦 GenAI Banking Policy & Compliance Copilot

A **Retrieval Augmented Generation (RAG)** system that lets compliance officers ask
natural language questions and receive accurate, cited answers from internal policy
documents — powered by **Microsoft Phi-2 (local SLM)** and **FAISS**.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  STREAMLIT UI  (app.py)                   │
│  Upload PDFs | Chat Interface | Citations | Index Status  │
└─────────────────────┬────────────────────────────────────┘
                      │ user question
                      ▼
┌──────────────────────────────────────────────────────────┐
│              RAG PIPELINE  (src/rag_pipeline.py)          │
│  1. Embed Query                                          │
│  2. FAISS Similarity Search  →  Top-K chunks            │
│  3. Build Context String                                 │
│  4. Phi-2 Generates Answer (strict RAG prompt)          │
│  5. Return Answer + Source Citations                     │
└──────┬───────────────────────────────┬───────────────────┘
       │                               │
┌──────┴───────────────┐   ┌───────────┴──────────────────┐
│  EMBEDDING ENGINE     │   │  VECTOR STORE (FAISS)        │
│  all-MiniLM-L6-v2    │   │  data/vectorstore/           │
│  (local, no API key) │   │  (persisted to disk)         │
└──────────────────────┘   └──────────────────────────────┘
       ▲
       │ chunks
┌──────┴───────────────────────┐
│  DOCUMENT PROCESSOR          │
│  PDF → pages → clean → split │
│  (pypdf / pdfplumber)        │
└──────────────────────────────┘
       ▲
       │ PDFs
┌──────┴──────────┐
│  data/uploads/  │
└─────────────────┘
```

---

## 📁 Project Structure

```
compliance_copilot/
│
├── app.py                         # Streamlit UI – run this
├── ingest.py                      # CLI batch-ingestion script
├── requirements.txt
├── .env.example                   # Config template
│
├── src/
│   ├── config.py                  # All settings + env-var overrides
│   ├── document_processor.py      # FR1, FR2 – PDF load & chunking
│   ├── vector_store.py            # FR3, FR4, FR6 – Embeddings + FAISS
│   ├── llm_engine.py              # FR7 – Phi-2 local inference
│   └── rag_pipeline.py            # FR5, FR7, FR8 – RAG orchestration
│
├── tests/
│   ├── test_document_processor.py # Unit tests – cleaning & chunking
│   ├── test_vector_store.py       # Unit tests – FAISS ops
│   └── test_rag_pipeline.py       # Integration tests (LLM mocked)
│
└── data/
    ├── uploads/                   # Drop PDF policy files here
    └── vectorstore/               # Auto-created FAISS index
```

---

## ⚙️ Setup

### 1. Prerequisites
- Python 3.10+
- ~6 GB free disk space (Phi-2 weights, downloaded on first use)
- 8 GB RAM minimum (16 GB recommended)

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure (optional)
```bash
cp .env.example .env
# Edit .env to change model, chunk sizes, etc.
```

---

## 🚀 Running the App

### Streamlit UI
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**Workflow:**
1. Use the **sidebar** to upload PDF policy documents
2. Wait for indexing to complete
3. Type your question in the chat box
4. Expand **Sources** under each answer to see citations

### CLI Pre-ingestion
```bash
python ingest.py                      # index all PDFs in data/uploads/
python ingest.py path/to/file.pdf     # index a specific file
python ingest.py --reset              # clear and rebuild index
python ingest.py --status             # show index stats
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

> Tests use a **mock LLM** — Phi-2 is never loaded during testing.
> The embedding model (~90 MB) downloads on first test run.

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `PHI2_MODEL_NAME` | `microsoft/phi-2` | HuggingFace model ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |
| `MAX_NEW_TOKENS` | `512` | Max LLM output tokens |
| `TEMPERATURE` | `0.1` | Lower = more factual answers |

---

## 📋 BRD Requirements Traceability

| Req | Requirement | Implemented In |
|-----|-------------|----------------|
| FR1 | Document Upload (PDF) | `app.py` uploader + `document_processor.save_uploaded_file()` |
| FR2 | Text extraction + chunking | `document_processor.load_and_chunk_pdf()` |
| FR3 | Embedding generation | `vector_store.LocalEmbeddings` (all-MiniLM-L6-v2) |
| FR4 | FAISS vector storage | `vector_store.ComplianceVectorStore` |
| FR5 | Natural language Q&A UI | `app.py` chat interface |
| FR6 | Context retrieval | `vector_store.similarity_search()` |
| FR7 | AI response generation | `llm_engine.Phi2LLM` + RAG prompt template |
| FR8 | Source citations | `rag_pipeline.CitedSource` + UI expander |
| NFR | Response < 5 seconds | Validated in `test_rag_pipeline` |
| NFR | Local storage only | `data/uploads/` + `data/vectorstore/` |

---

## 💡 Example Queries

```
What documents are required for KYC verification?
What is the AML reporting threshold?
What are the penalties for non-compliance with AML regulations?
Describe the risk assessment process for loan approval.
What is the procedure for filing a Suspicious Activity Report?
How often must customer due diligence be renewed?
```

---

## ⚠️ Phi-2 First-Run Note

On the **first question**, Phi-2 weights (~5.5 GB) are downloaded from
HuggingFace Hub automatically. This takes a few minutes once.
Subsequent runs load from cache instantly.

To pre-download manually:
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('microsoft/phi-2', trust_remote_code=True)
print('Phi-2 downloaded successfully.')
"
```

---

## 🗓️ Project Timeline (per BRD §15)

| Phase | Duration | Status |
|-------|----------|--------|
| Requirement Analysis | 2 days | ✅ Done |
| System Design | 2 days | ✅ Done |
| Development | 1 day | ✅ Done |
| Testing | 1 day | ✅ Done |
| Demo & Evaluation | 1 day | 🔲 Pending |

---

*Built by Akshay & Tejas — GenAI Banking Policy & Compliance Copilot POC*

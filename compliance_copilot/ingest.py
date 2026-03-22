"""
ingest.py  – CLI batch-ingestion script
Usage:
    python ingest.py                   # index all PDFs in data/uploads/
    python ingest.py path/to/file.pdf  # index a specific file
    python ingest.py --reset           # clear index first
    python ingest.py --status          # show stats only
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_ingest(pdf_paths, reset=False):
    from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, VECTORSTORE_DIR
    from src.document_processor import load_and_chunk_pdf
    from src.vector_store import ComplianceVectorStore
    store = ComplianceVectorStore(VECTORSTORE_DIR, EMBEDDING_MODEL)
    if reset:
        logger.info("Resetting vector store...")
        store.reset()
    for pdf in pdf_paths:
        if not pdf.exists():
            logger.warning("Not found, skipping: %s", pdf)
            continue
        logger.info("Ingesting: %s", pdf.name)
        chunks = load_and_chunk_pdf(pdf, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        store.add_documents(chunks)
        logger.info("  -> %d chunks added", len(chunks))
    logger.info("Done. Total chunks in store: %d", store.document_count())


def show_status():
    from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR
    from src.vector_store import ComplianceVectorStore
    store = ComplianceVectorStore(VECTORSTORE_DIR, EMBEDDING_MODEL)
    count = store.document_count()
    print(f"\nVector store : {VECTORSTORE_DIR}")
    print(f"Total chunks : {count}")
    print(f"Status       : {'Populated' if count else 'Empty'}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--reset",  action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        show_status(); return
    from src.config import UPLOAD_DIR
    pdf_paths = [Path(f) for f in args.files] if args.files else sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found. Copy files to data/uploads/ or pass paths directly.")
        return
    run_ingest(pdf_paths, reset=args.reset)

if __name__ == "__main__":
    main()

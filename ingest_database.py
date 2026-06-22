import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========================================
# CONFIGURATION
# ========================================

# Must match the collection name used in rag-pipeline.py
COLLECTION_NAME = "knowledge_base"
CHROMA_DB_PATH = "./chroma_db"
# Tracks which files are ingested and their hashes for incremental updates
STATE_FILE = "./chroma_db/.ingest_state.json"

KNOWLEDGE_DOCS_DIR = Path(__file__).resolve().parent / "knowledge-docs"

# Larger chunks give better context per retrieval result
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ========================================
# SECTION 1: FILE STATE TRACKING
# ========================================

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for reliable change detection."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_ingest_state() -> Dict:
    """Load ingestion state from disk. Returns empty dict on first run."""
    state_path = Path(STATE_FILE)
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ingest_state(state: Dict) -> None:
    """Persist ingestion state to disk after every successful run."""
    state_path = Path(STATE_FILE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ========================================
# SECTION 2: DOCUMENT LOADING & CHUNKING
# ========================================

def load_pdf_pages(file_path: Path) -> List[Dict]:
    """Extract text page by page from a single PDF file."""
    documents = []
    doc_base_id = file_path.stem.lower().replace(" ", "_")
    title = file_path.stem

    # Use the immediate parent folder relative to knowledge-docs as the category.
    # Files directly under knowledge-docs/ get category "general".
    try:
        rel = file_path.relative_to(KNOWLEDGE_DOCS_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
    except ValueError:
        category = "general"

    reader = pypdf.PdfReader(str(file_path))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        documents.append({
            "id": f"{doc_base_id}_p{page_num + 1}",
            "title": title,
            "content": text.strip(),
            "category": category,
            "source_path": str(file_path),
            "page": page_num + 1,
        })

    return documents


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Split page-level documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        for i, chunk in enumerate(text_splitter.split_text(doc["content"])):
            all_chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "title": doc["title"],
                "content": chunk,
                "category": doc["category"],
                "source_doc": doc["id"],
                "source_path": doc["source_path"],
                "page": doc.get("page", 0),
            })

    return all_chunks


# ========================================
# SECTION 3: CHROMADB OPERATIONS
# ========================================

def get_chromadb_collection() -> chromadb.Collection:
    """Initialize ChromaDB client and return (or create) the collection."""
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def remove_file_chunks(collection: chromadb.Collection, chunk_ids: List[str]) -> None:
    """Delete all stored chunks belonging to a specific file."""
    if chunk_ids:
        collection.delete(ids=chunk_ids)


def add_chunks_to_collection(collection: chromadb.Collection, chunks: List[Dict]) -> None:
    """
    Upsert document chunks into ChromaDB.
    ChromaDB auto-embeds documents using its built-in all-MiniLM-L6-v2 model.
    Using upsert makes re-runs safe and idempotent.
    """
    if not chunks:
        return

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "title": c["title"],
                "category": c["category"],
                "source_doc": c["source_doc"],
                "source_path": c["source_path"],
                "page": c["page"],
            }
            for c in chunks
        ],
    )


# ========================================
# SECTION 4: INCREMENTAL INGESTION LOGIC
# ========================================

def scan_pdf_files(docs_dir: Path) -> Dict[str, str]:
    """
    Scan directory for all PDF files.
    Returns a dict mapping a stable file key (relative path) to absolute path string.
    """
    found = {}
    for file_path in sorted(docs_dir.rglob("*.pdf")):
        key = str(file_path.relative_to(docs_dir)).replace("\\", "/")
        found[key] = str(file_path)
    return found


def classify_files(
    current_files: Dict[str, str],
    state: Dict,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare current PDF files against the persisted state.
    Returns (new_keys, modified_keys, deleted_keys).
    File identity is based on content hash, not mtime, for reliability.
    """
    new_keys, modified_keys, deleted_keys = [], [], []

    for key, path in current_files.items():
        if key not in state:
            new_keys.append(key)
        else:
            current_hash = compute_file_hash(path)
            if current_hash != state[key].get("file_hash", ""):
                modified_keys.append(key)

    for key in state:
        if key not in current_files:
            deleted_keys.append(key)

    return new_keys, modified_keys, deleted_keys


# ========================================
# SECTION 5: MAIN INGESTION ORCHESTRATOR
# ========================================

def run_ingestion() -> None:
    """
    Full ingestion pipeline suitable for both first-run and daily incremental updates.

    First run  -> ingest every PDF found under knowledge-docs/.
    Daily run  -> detect new / modified / deleted files and update ChromaDB accordingly.
    No changes -> exit immediately without touching the database.
    """
    print("=" * 60)
    print("   RAG Chatbot - Database Ingestion")
    print("=" * 60)
    start_time = time.time()

    # --- 1. Load previous state ---
    state = load_ingest_state()
    is_first_run = len(state) == 0
    run_label = "First run - ingesting all documents" if is_first_run else "Incremental run - detecting changes"
    print(f"\n{run_label}")

    # --- 2. Scan current PDF files ---
    print(f"\nScanning: {KNOWLEDGE_DOCS_DIR}")
    current_files = scan_pdf_files(KNOWLEDGE_DOCS_DIR)

    if not current_files:
        print(f"WARNING: No PDF files found in {KNOWLEDGE_DOCS_DIR}. Nothing to ingest.")
        return

    print(f"Found {len(current_files)} PDF file(s):")
    for key in current_files:
        print(f"  - {key}")

    # --- 3. Classify files ---
    new_keys, modified_keys, deleted_keys = classify_files(current_files, state)

    if not new_keys and not modified_keys and not deleted_keys:
        print("\nNo changes detected. ChromaDB is already up to date.")
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s")
        return

    # --- 4. Connect to ChromaDB ---
    print(f"\nConnecting to ChromaDB at: {CHROMA_DB_PATH}")
    collection = get_chromadb_collection()
    print(f"Collection '{COLLECTION_NAME}' - {collection.count()} existing chunks")

    # --- 5. Remove deleted files ---
    if deleted_keys:
        print(f"\nRemoving {len(deleted_keys)} deleted file(s)...")
        for key in deleted_keys:
            chunk_ids = state[key].get("chunk_ids", [])
            remove_file_chunks(collection, chunk_ids)
            del state[key]
            print(f"  Removed: {key} ({len(chunk_ids)} chunks)")

    # --- 6. Clear stale chunks for modified files before re-ingesting ---
    if modified_keys:
        print(f"\nRe-ingesting {len(modified_keys)} modified file(s)...")
        for key in modified_keys:
            chunk_ids = state[key].get("chunk_ids", [])
            remove_file_chunks(collection, chunk_ids)
            print(f"  Cleared stale chunks for: {key}")

    # --- 7. Ingest new and modified files ---
    files_to_ingest = {k: current_files[k] for k in new_keys + modified_keys}
    if files_to_ingest:
        print(f"\nIngesting {len(files_to_ingest)} file(s)...")

    for key, path in files_to_ingest.items():
        file_path = Path(path)
        print(f"\n  Processing: {file_path.name}")

        pages = load_pdf_pages(file_path)
        if not pages:
            print(f"  WARNING: No readable text in {file_path.name}, skipping.")
            continue

        chunks = chunk_documents(pages)
        add_chunks_to_collection(collection, chunks)

        state[key] = {
            "source_path": path,
            "file_hash": compute_file_hash(path),
            "chunk_ids": [c["id"] for c in chunks],
            "pages": len(pages),
            "chunks": len(chunks),
        }
        print(f"  OK {file_path.name}: {len(pages)} pages -> {len(chunks)} chunks")

    # --- 8. Persist updated state ---
    save_ingest_state(state)

    # --- 9. Summary ---
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  New: {len(new_keys)}  |  Modified: {len(modified_keys)}  |  Deleted: {len(deleted_keys)}")
    print(f"  Total chunks in ChromaDB: {collection.count()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_ingestion()

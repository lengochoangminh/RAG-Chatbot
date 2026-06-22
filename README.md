# RAG Chatbot — Personal Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in your own PDF documents. It combines a local ChromaDB vector store for semantic retrieval with Azure AI Foundry (OpenAI) for response generation.

---

## Framework Overview

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Query Reformulation  │  (LLM fixes typos, resolves
             │  (Azure AI Foundry)   │   follow-ups from history)
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │   ChromaDB Retrieval  │  Top-K cosine-similarity search
             │  (all-MiniLM-L6-v2)   │  over ingested PDF chunks
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Prompt Augmentation  │  System prompt + retrieved
             │                       │  excerpts + chat history
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Response Generation  │  Azure AI Foundry LLM
             │  (Azure AI Foundry)   │  answers strictly from context
             └───────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|---|---|---|
| Vector store | ChromaDB (local, persistent) | Zero-infrastructure, file-based |
| Embedding model | `all-MiniLM-L6-v2` (via ChromaDB built-in) | Same model at ingest & query time — no mismatch |
| Chunking | `RecursiveCharacterTextSplitter` (500 / 100 overlap) | Balances context per chunk vs. retrieval precision |
| Change detection | SHA-256 file hash | Reliable incremental re-ingestion on file edits |
| LLM | Azure AI Foundry (configurable deployment) | Grounded, no hallucination beyond provided excerpts |

---

## Project Structure

```
RAG-Chatbot/
├── chatbot.py              # Interactive chat loop (main entry point)
├── ingest_database.py      # PDF ingestion pipeline → ChromaDB
├── requirement.txt         # Python dependencies
├── .env                    # Environment variables (not committed)
├── knowledge-docs/         # Drop your PDF files here
│   └── <category>/         # Optional sub-folders become the "category" metadata
│       └── document.pdf
└── chroma_db/              # Auto-created by ingest_database.py
    ├── chroma.sqlite3
    └── .ingest_state.json  # Tracks file hashes for incremental updates
```

---

## Prerequisites

- Python 3.10+
- An **Azure AI Foundry** project with a deployed chat model
- PDF documents to query placed under `knowledge-docs/`

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd RAG-Chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirement.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
AZURE_PROJECT_ENDPOINT=https://<your-hub>.services.ai.azure.com/api/projects/<your-project>
MODEL_DEPLOYMENT_NAME=<your-model-deployment-name>
```

Azure authentication uses `DefaultAzureCredential`. Run `az login` (Azure CLI) before starting the chatbot if you are not running inside a managed-identity environment.

---

## Usage

### Step 1 — Ingest your documents

Place PDF files anywhere under `knowledge-docs/` (sub-folders are supported and become the `category` metadata field).

```bash
python ingest_database.py
```

- **First run**: ingests every PDF found.
- **Subsequent runs**: only processes new, modified, or deleted files (incremental).
- Re-run any time you add or update documents.

### Step 2 — Start the chatbot

```bash
python chatbot.py
```

#### In-chat commands

| Command | Action |
|---|---|
| `quit` / `exit` | Exit the chatbot |
| `clear` | Reset conversation history |

---

## Configuration

Key constants at the top of each script can be adjusted:

| File | Constant | Default | Description |
|---|---|---|---|
| `chatbot.py` | `TOP_K` | `5` | Number of chunks retrieved per query |
| `chatbot.py` | `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to consider relevant |
| `ingest_database.py` | `CHUNK_SIZE` | `500` | Characters per chunk |
| `ingest_database.py` | `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| Both | `COLLECTION_NAME` | `knowledge_base` | Must match between both scripts |

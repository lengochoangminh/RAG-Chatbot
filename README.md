# RAG Chatbot — Personal Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in two knowledge sources: your own PDF documents (via ChromaDB) and a live Confluence page (via an Atlassian MCP server). It uses Azure AI Foundry (OpenAI) for response generation.

---

## Framework Overview

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Query Reformulation  │  LLM fixes typos, resolves
             │  (Azure AI Foundry)   │  follow-ups from chat history
             └───────────┬───────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
  ┌─────────────────────┐  ┌──────────────────────┐
  │  ChromaDB Retrieval │  │  Confluence Page     │
  │  (PDF chunks,       │  │  (fetched live via   │
  │  all-MiniLM-L6-v2)  │  │  MCP Atlassian)      │
  └──────────┬──────────┘  └──────────┬───────────┘
             │                        │
             └──────────┬─────────────┘
                        ▼
            ┌───────────────────────┐
            │  Prompt Augmentation  │  System prompt + both sources
            │                       │  + chat history (last 20 turns)
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Response Generation  │  LLM tool-call loop; cites
            │  (Azure AI Foundry)   │  section or document + page
            └───────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|---|---|---|
| Vector store | ChromaDB (local, persistent) | Zero-infrastructure, file-based |
| Embedding model | `all-MiniLM-L6-v2` (via ChromaDB built-in) | Same model at ingest & query time — no mismatch |
| Chunking | `RecursiveCharacterTextSplitter` (500 / 100 overlap) | Balances context per chunk vs. retrieval precision |
| Change detection | SHA-256 file hash | Reliable incremental re-ingestion on file edits |
| Live knowledge source | Atlassian MCP server (`mcp-atlassian`) | Fetches Confluence pages at startup without manual export |
| Tool execution | MCP tool-call loop (async) | LLM can invoke Atlassian tools and incorporate results before answering |
| LLM | Azure AI Foundry (configurable deployment) | Grounded, no hallucination beyond provided sources |

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
- A Confluence (Server/Data Center) instance accessible via a personal access token
- `mcp-atlassian` CLI installed and available on `PATH` (see [sooperset/mcp-atlassian](https://github.com/sooperset/mcp-atlassian))

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
# Azure AI Foundry
AZURE_PROJECT_ENDPOINT=https://<your-hub>.services.ai.azure.com/api/projects/<your-project>
MODEL_DEPLOYMENT_NAME=<your-model-deployment-name>

# Confluence (Server / Data Center)
CONFLUENCE_BASE_URL=https://confluence.example.com
CONFLUENCE_TOKEN=<your-confluence-personal-access-token>

# Jira (optional — required only if using Jira MCP tools)
JIRA_BASE_URL=https://jira.example.com
JIRA_TOKEN=<your-jira-personal-access-token>
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

On startup the chatbot will:
1. Connect to ChromaDB and report the number of available chunks.
2. Authenticate with Azure AI Foundry.
3. Connect to the `mcp-atlassian` MCP server and fetch the configured Confluence page.
4. Enter the interactive chat loop.

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
| `chatbot.py` | `ONBOARDING_PAGE_ID` | `128180006` | Confluence page ID to fetch at startup |
| `chatbot.py` | `ONBOARDING_PAGE_TITLE` | _(see file)_ | Display name for the Confluence source |
| `ingest_database.py` | `CHUNK_SIZE` | `500` | Characters per chunk |
| `ingest_database.py` | `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| Both | `COLLECTION_NAME` | `knowledge_base` | Must match between both scripts |

---

## Dependencies

| Package | Purpose |
|---|---|
| `chromadb` | Local vector store with built-in embedding |
| `pypdf` | PDF text extraction |
| `langchain-text-splitters` | Recursive chunking |
| `azure-ai-projects` / `azure-identity` | Azure AI Foundry client & auth |
| `mcp` | Model Context Protocol client SDK |
| `mcp-atlassian` | MCP server that wraps Confluence & Jira APIs |
| `python-dotenv` | `.env` file loading |

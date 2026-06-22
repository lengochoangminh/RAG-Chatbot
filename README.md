# RAG Chatbot — Personal Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in two knowledge sources: your own PDF documents (via ChromaDB) and a live Confluence page (via an Atlassian MCP server). It uses Azure AI Foundry (OpenAI) for response generation.

Available in two modes:
- **`chatbot.py`** — interactive terminal chatbot
- **`chatbot-v2.py`** — Slack bot; mention `@SonicBot` in any channel to ask a question

---

## Problem Statement

Every time a new hire joins the team, a senior engineer or team lead must invest significant time walking them through onboarding: explaining processes, pointing to the right documents, and answering the same recurring questions — often repeatedly across different hires.

This creates friction for both sides:
- **For the team**: senior members are pulled away from productive work to answer questions that are already documented somewhere.
- **For the new hire**: finding the right information is slow and depends on who is available, leading to delays and inconsistent answers.

### How this chatbot helps

The RAG Chatbot gives new hires **instant, accurate, self-service access** to the team's knowledge base — no need to interrupt a colleague or manually search through Confluence pages and PDFs.

| Without RAG Chatbot | With RAG Chatbot |
|---|---|
| Ask a senior → wait for a reply | Ask `@SonicBot` → get an answer in seconds |
| Answer quality depends on who you ask | Answers are grounded in the official documents |
| Same questions answered over and over | Knowledge is always available, 24/7 |
| New hires feel unsure where to look | New hires can explore confidently on their own |
| Must ask in the team's primary language | Ask in **any language** — the chatbot understands and replies accordingly |

New hires ask questions in plain language, **in whichever language they are most comfortable with** — the chatbot automatically retrieves the most relevant content across the onboarding checklist, policy PDFs, and benefit documents, and cites the exact source so they can read further if needed.

---

## Framework Overview

```
┌──────────────────────────────────────────────────────────────┐
│          User Query  (terminal input  OR  Slack @mention)    │
└──────────────────────────────┬───────────────────────────────┘
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
| Slack interface | `slack-bolt` + Socket Mode | No public URL needed; works behind firewalls |
| Per-user history | In-memory dict (thread-safe lock) | Each Slack user gets an independent conversation context |
| Slack message size | Chunked at 3 800 chars | Stays within Slack's 4 000-char hard limit |

---

## Project Structure

```
RAG-Chatbot/
├── chatbot.py              # Interactive terminal chatbot
├── chatbot-v2.py           # Slack bot (Socket Mode)
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
- _(Slack bot only)_ A Slack app with **Socket Mode** enabled, the `app_mentions:read` and `chat:write` scopes, and both a Bot User OAuth Token (`xoxb-…`) and an App-Level Token (`xapp-…`)

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

# Slack bot (chatbot-v2.py only)
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
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

---

### Step 2 — Slack bot

```bash
python chatbot-v2.py
```

On startup the bot will:
1. Connect to ChromaDB.
2. Authenticate with Azure AI Foundry.
3. Pre-fetch the Confluence page via `mcp-atlassian`.
4. Connect to Slack via **Socket Mode** (no public URL required).

Once running, **mention `@SonicBot`** in any channel:

| Slack message | Action |
|---|---|
| `@SonicBot <your question>` | Answer from ChromaDB + Confluence |
| `@SonicBot clear history` | Reset your personal conversation history |
| `@SonicBot help` | Show available commands |

> Each Slack user has their own isolated conversation history (last 10 turns). Long answers are automatically split to respect Slack's 4 000-character message limit.

---

### Step 3 — Ask Questions

The chatbot is grounded in the following knowledge sources:

| Source | Content |
|---|---|
| Confluence page | New Employee Onboarding Checklist (English Version) |
| PDF | 2026 Company Holiday Schedule |
| PDF | 2026 Employee Benefit Packet |
| PDF | Employee Referral Program Policy & Procedures |

Ask your question naturally — the chatbot will automatically search the most relevant source(s) and cite the document name, page, or Confluence section in its answer.

#### Example questions

**Onboarding**
- *"What are the steps I need to complete in my first week?"*
- *"What IT access requests should I submit on day one?"*
- *"Who do I contact to get my employee badge?"*

**Company holidays**
- *"What are the public holidays for 2026?"*
- *"Is there a day off around Thanksgiving?"*

**Benefits**
- *"What health insurance plans are available?"*
- *"How do I enroll in the 401(k) plan?"*
- *"What is the annual leave entitlement for new employees?"*

**Referral program**
- *"How does the employee referral bonus work?"*
- *"What is the process to refer a candidate?"*
- *"Are there any restrictions on who I can refer?"*

#### Demo

<img src=".\screenshots\RAG-Demo.jpg" width="700">

---


## Configuration

Key constants at the top of each script can be adjusted:

| File | Constant | Default | Description |
|---|---|---|---|
| `chatbot.py` / `chatbot-v2.py` | `TOP_K` | `5` | Number of chunks retrieved per query |
| `chatbot.py` / `chatbot-v2.py` | `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to consider relevant |
| `chatbot.py` / `chatbot-v2.py` | `ONBOARDING_PAGE_ID` | `128180006` | Confluence page ID to fetch at startup |
| `chatbot.py` / `chatbot-v2.py` | `ONBOARDING_PAGE_TITLE` | _(see file)_ | Display name for the Confluence source |
| `chatbot-v2.py` | `MAX_HISTORY_TURNS` | `10` | Per-user turns kept in memory (each turn = 2 messages) |
| `ingest_database.py` | `CHUNK_SIZE` | `500` | Characters per chunk |
| `ingest_database.py` | `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| All | `COLLECTION_NAME` | `knowledge_base` | Must match across all scripts |

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
| `slack-bolt` | Slack app framework (event handling, Socket Mode) |
| `slack-sdk` | Slack Web API client (posting messages) |
| `python-dotenv` | `.env` file loading |

"""
SonicBot — Slack-integrated Personal Assistant
===============================================
Slack bot that answers questions using documents stored in ChromaDB and a
Confluence page fetched via MCP.  Mention @SonicBot in any channel to ask
a question.  Each Slack user gets their own conversation history.

Prerequisites:
  1. Run ingest_database.py to populate ChromaDB.
  2. Create a Slack app with Socket Mode enabled and add the following env vars
     to your .env file:
       SLACK_BOT_TOKEN   - Bot User OAuth Token  (xoxb-…)
       SLACK_APP_TOKEN   - App-Level Token        (xapp-…)
  3. Also required (same as chatbot.py):
       AZURE_PROJECT_ENDPOINT, MODEL_DEPLOYMENT_NAME,
       CONFLUENCE_BASE_URL / CONFLUENCE_TOKEN (or JIRA_BASE_URL / JIRA_TOKEN)
"""

import os
import sys
import asyncio
import json
import threading
from typing import Any, List, Dict

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import chromadb
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

# ========================================
# CONFIGURATION
# ========================================

COLLECTION_NAME = "knowledge_base"
CHROMA_DB_PATH = "./chroma_db"

SIMILARITY_THRESHOLD = 0.3
TOP_K = 5

ONBOARDING_PAGE_ID = "128180006"
ONBOARDING_PAGE_TITLE = "01. New Employee Onboarding Checklist (English Version)"

# Maximum recent turns kept in each user's history (each turn = 2 messages)
MAX_HISTORY_TURNS = 10

load_dotenv()

SYSTEM_PROMPT = """\
You are a helpful assistant with two sources of information:
1. A Confluence page (New Employee Onboarding Checklist) — provided below.
2. Local document excerpts retrieved from a knowledge base — also provided below.

Rules:
- Answer the question using information from EITHER source.
- If the Confluence page covers the topic, cite the relevant section number.
- If the local documents cover the topic, cite the source document name and page.
- If NEITHER source contains relevant information, say so clearly.
- Do NOT make up information that is not in the provided sources.
"""

REFORMULATE_PROMPT = """\
Rewrite the user's latest message into a clear, self-contained search query \
that can be used to find relevant documents. \
Fix any typos or misspellings. \
Incorporate context from the conversation history so the query stands alone. \
Return ONLY the improved search query, nothing else."""

# ========================================
# ENVIRONMENT VALIDATION
# ========================================

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise ValueError(
        f"Missing Slack tokens — "
        f"SLACK_BOT_TOKEN={'set' if SLACK_BOT_TOKEN else 'MISSING'}, "
        f"SLACK_APP_TOKEN={'set' if SLACK_APP_TOKEN else 'MISSING'}"
    )

AZURE_PROJECT_ENDPOINT = os.getenv("AZURE_PROJECT_ENDPOINT")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
if not AZURE_PROJECT_ENDPOINT or not MODEL_DEPLOYMENT_NAME:
    raise ValueError(
        f"Missing Azure config — "
        f"AZURE_PROJECT_ENDPOINT={'set' if AZURE_PROJECT_ENDPOINT else 'MISSING'}, "
        f"MODEL_DEPLOYMENT_NAME={'set' if MODEL_DEPLOYMENT_NAME else 'MISSING'}"
    )

# ========================================
# SLACK APP
# ========================================

slack_app = App(token=SLACK_BOT_TOKEN)
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# ========================================
# AZURE AI FOUNDRY CLIENT
# ========================================

project = AIProjectClient(
    endpoint=AZURE_PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)
openai_client = project.get_openai_client()
model = MODEL_DEPLOYMENT_NAME

# ========================================
# CHROMADB
# ========================================

def get_collection() -> chromadb.Collection:
    """Connect to ChromaDB and return the knowledge_base collection."""
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = db_client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found.")
        print("Run  python ingest_database.py  first to populate ChromaDB.")
        sys.exit(1)

    if collection.count() == 0:
        print(f"ERROR: Collection '{COLLECTION_NAME}' is empty.")
        print("Run  python ingest_database.py  first to populate ChromaDB.")
        sys.exit(1)

    return collection


def search(collection: chromadb.Collection, query: str) -> List[Dict]:
    """Search ChromaDB for chunks relevant to the query."""
    results = collection.query(query_texts=[query], n_results=TOP_K)

    search_results = []
    for doc_id, distance, content, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        similarity = 1 - distance
        search_results.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "similarity": similarity,
        })

    return search_results


# ========================================
# MCP ATLASSIAN TOOLS
# ========================================

def _build_mcp_env() -> dict:
    """Build subprocess env for mcp-atlassian, mapping .env names to what it expects."""
    env = {**os.environ}
    if os.getenv("JIRA_BASE_URL"):
        env["JIRA_URL"] = os.getenv("JIRA_BASE_URL")
    if os.getenv("JIRA_TOKEN"):
        env["JIRA_PERSONAL_TOKEN"] = os.getenv("JIRA_TOKEN")
    if os.getenv("CONFLUENCE_BASE_URL"):
        env["CONFLUENCE_URL"] = os.getenv("CONFLUENCE_BASE_URL")
    if os.getenv("CONFLUENCE_TOKEN"):
        env["CONFLUENCE_PERSONAL_TOKEN"] = os.getenv("CONFLUENCE_TOKEN")
    return env


async def _fetch_confluence_page() -> str:
    """Start an MCP session, fetch the onboarding page, then close the session."""
    server_params = StdioServerParameters(
        command="mcp-atlassian",
        args=[],
        env=_build_mcp_env(),
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "confluence_get_page", {"page_id": ONBOARDING_PAGE_ID}
                )
                return "\n".join(
                    block.text if hasattr(block, "text") else str(block)
                    for block in result.content
                )
    except Exception as exc:
        print(f"[startup] Failed to fetch Confluence page: {exc}")
        return ""


# ========================================
# QUERY REFORMULATION
# ========================================

def reformulate_query(raw_query: str, chat_history: List[Dict]) -> str:
    """Use the LLM to produce a clean, self-contained search query."""
    messages = [{"role": "system", "content": REFORMULATE_PROMPT}]
    messages.extend(chat_history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": raw_query})

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# ========================================
# PROMPT AUGMENTATION
# ========================================

def build_augmented_messages(
    query: str,
    chat_history: List[Dict],
    confluence_page: str = "",
    search_results: List[Dict] = None,
) -> List[Dict]:
    """Build the messages list for the LLM call."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if confluence_page:
        messages.append({
            "role": "system",
            "content": f"CONFLUENCE PAGE: {ONBOARDING_PAGE_TITLE}\n\n{confluence_page}",
        })

    if search_results:
        context_parts = []
        for i, r in enumerate(search_results, 1):
            page = r["metadata"].get("page", "?")
            title = r["metadata"].get("title", "Unknown")
            context_parts.append(
                f"[Source {i}: {title}, Page {page}]\n{r['content']}"
            )
        messages.append({
            "role": "system",
            "content": f"LOCAL DOCUMENT EXCERPTS:\n\n{'\n\n'.join(context_parts)}",
        })

    # Keep last MAX_HISTORY_TURNS turns (each turn = user + assistant message)
    messages.extend(chat_history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": query})

    return messages


# ========================================
# RESPONSE GENERATION
# ========================================

def generate_response(messages: List[Dict]) -> str:
    """Call the LLM and return the assistant reply."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


# ========================================
# GLOBAL STATE
# ========================================

# Populated at startup
collection: chromadb.Collection = None
onboarding_content: str = ""

# Per-user conversation history:  { slack_user_id: [{"role": ..., "content": ...}, ...] }
user_histories: Dict[str, List[Dict]] = {}
_histories_lock = threading.Lock()

# Deduplicate Slack event retries
_processed_event_ids: set = set()

# ========================================
# SLACK HELPERS
# ========================================

def _chunks(text: str, size: int = 3800) -> List[str]:
    """Split text into Slack-safe chunks (Slack hard limit: 4 000 chars)."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def _get_user_history(user_id: str) -> List[Dict]:
    with _histories_lock:
        if user_id not in user_histories:
            user_histories[user_id] = []
        return user_histories[user_id]


def _append_history(user_id: str, role: str, content: str):
    with _histories_lock:
        if user_id not in user_histories:
            user_histories[user_id] = []
        user_histories[user_id].append({"role": role, "content": content})


def _clear_history(user_id: str):
    with _histories_lock:
        user_histories[user_id] = []


# ========================================
# CORE RAG HANDLER (runs in background thread)
# ========================================

def _handle_rag_query(channel: str, user: str, question: str):
    """Retrieve context, generate a response, and post it to Slack."""
    history = _get_user_history(user)

    # Reformulate the query for better retrieval
    try:
        search_query = reformulate_query(question, history)
    except Exception:
        search_query = question

    if search_query.lower() != question.lower():
        print(f"[rag] user={user} search_query={search_query!r}")

    # Retrieve relevant chunks from ChromaDB
    results = search(collection, search_query)

    # Build the augmented prompt
    messages = build_augmented_messages(
        question,
        history,
        confluence_page=onboarding_content,
        search_results=results,
    )

    # Call the LLM
    try:
        answer = generate_response(messages)
    except Exception as exc:
        print(f"[rag-error] {exc}")
        slack_client.chat_postMessage(
            channel=channel,
            text=f"<@{user}> :x: Error generating response: {exc}",
        )
        return

    # Persist turn in history
    _append_history(user, "user", question)
    _append_history(user, "assistant", answer)

    # Post the answer (split if over Slack's limit)
    for chunk in _chunks(f"<@{user}> {answer}"):
        slack_client.chat_postMessage(channel=channel, text=chunk)


# ========================================
# SLACK EVENT HANDLERS
# ========================================

@slack_app.event("app_mention")
def handle_mention(event, say):
    """Route @SonicBot mentions: answer questions or handle special commands."""
    # Deduplicate — Slack retries delivery if ACK is slow
    event_key = event.get("client_msg_id") or event.get("ts", "")
    if event_key in _processed_event_ids:
        print(f"[mention] duplicate event skipped: {event_key!r}")
        return
    _processed_event_ids.add(event_key)
    if len(_processed_event_ids) > 500:
        _processed_event_ids.clear()

    user = event["user"]
    channel = event["channel"]
    text = event.get("text", "")

    # Strip the bot mention prefix  (<@UXXXXXX> …)
    question = text.split(">", 1)[-1].strip()

    if not question:
        slack_client.chat_postMessage(
            channel=channel,
            text=(
                f"<@{user}> Hi! Ask me anything about the knowledge base. "
                "Type `help` for available commands."
            ),
        )
        return

    lowered = question.lower().strip()

    # --- Command: help ---
    if lowered == "help":
        slack_client.chat_postMessage(
            channel=channel,
            text=(
                "*SonicBot — Available commands:*\n"
                "• `<your question>` — ask anything from the knowledge base\n"
                "• `clear history` — reset your personal conversation history\n"
                "• `help` — show this message\n\n"
                f"_Knowledge base: {collection.count()} document chunks | "
                f"Confluence: {ONBOARDING_PAGE_TITLE}_"
            ),
        )
        return

    # --- Command: clear history ---
    if lowered in ("clear history", "clear", "reset"):
        _clear_history(user)
        slack_client.chat_postMessage(
            channel=channel,
            text=f"<@{user}> :white_check_mark: Your conversation history has been cleared.",
        )
        return

    # --- Default: RAG query ---
    slack_client.chat_postMessage(
        channel=channel,
        text=f"<@{user}> :mag: Searching knowledge base...",
    )

    threading.Thread(
        target=_handle_rag_query,
        args=(channel, user, question),
        daemon=True,
    ).start()


@slack_app.event("message")
def handle_message(event):
    """Ignore plain messages (only respond to @-mentions)."""
    pass


@slack_app.error
def global_error_handler(error, body, logger):
    """Log any unhandled Bolt errors to the console."""
    logger.exception(f"Unhandled error: {error}\nEvent body: {body}")


# ========================================
# STARTUP
# ========================================

def startup():
    """Initialise ChromaDB and pre-fetch the Confluence page before Slack starts."""
    global collection, onboarding_content

    print("=" * 60)
    print("  SonicBot — Slack Bot")
    print("=" * 60)

    # Connect to ChromaDB
    collection = get_collection()
    print(f"ChromaDB connected — {collection.count()} chunks available")

    # Pre-fetch the Confluence onboarding page via MCP
    print(f"Fetching Confluence page '{ONBOARDING_PAGE_TITLE}'...", end=" ", flush=True)
    onboarding_content = asyncio.run(_fetch_confluence_page())
    if onboarding_content:
        print(f"OK ({len(onboarding_content)} chars)")
    else:
        print("SKIPPED (page unavailable — only ChromaDB will be used)")

    print(f"LLM model : {model}")
    print(f"Sources   : Confluence + {collection.count()} ChromaDB chunks")
    print()


if __name__ == "__main__":
    startup()
    print("Starting SonicBot in Socket Mode...")
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()

"""
RAG Chatbot — Personal Assistant
=================================
Interactive chatbot that answers questions using documents stored in ChromaDB.
Run ingest_database.py first to populate the vector database, then launch this.
"""

import os
import sys
import asyncio
import json
from typing import Any, List, Dict
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import chromadb
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# ========================================
# CONFIGURATION
# ========================================

# Must match the values in ingest_database.py
COLLECTION_NAME = "knowledge_base"
CHROMA_DB_PATH = "./chroma_db"

SIMILARITY_THRESHOLD = 0.3
TOP_K = 5

# Single Confluence page used as the knowledge source
ONBOARDING_PAGE_ID = "128180006"
ONBOARDING_PAGE_TITLE = "01. New Employee Onboarding Checklist (English Version)"

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


# ========================================
# AZURE AI FOUNDRY CLIENT
# ========================================

def create_llm_client():
    """Initialize the Azure AI Foundry OpenAI client."""
    endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
    model = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not endpoint or not model:
        print("ERROR: Set AZURE_PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME in .env")
        sys.exit(1)

    project = AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )
    return project.get_openai_client(), model


# ========================================
# QUERY REFORMULATION
# ========================================

REFORMULATE_PROMPT = """\
Rewrite the user's latest message into a clear, self-contained search query \
that can be used to find relevant documents. \
Fix any typos or misspellings. \
Incorporate context from the conversation history so the query stands alone. \
Return ONLY the improved search query, nothing else."""


def reformulate_query(
    openai_client,
    model: str,
    raw_query: str,
    chat_history: List[Dict],
) -> str:
    """
    Use the LLM to turn the user's raw input into a clean, self-contained
    search query — fixing typos and resolving references from chat history.
    """
    messages = [{"role": "system", "content": REFORMULATE_PROMPT}]
    # Give the LLM recent conversation context so it can resolve follow-ups
    messages.extend(chat_history[-10:])
    messages.append({"role": "user", "content": raw_query})

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# ========================================
# CHROMADB RETRIEVAL
# ========================================

def get_collection() -> chromadb.Collection:
    """Connect to ChromaDB and return the knowledge_base collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
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
    """
    Search ChromaDB for chunks relevant to the query.
    Uses ChromaDB's built-in embedding (same model used at ingestion time)
    so there's no need to manually encode the query.
    """
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
# PROMPT AUGMENTATION
# ========================================

def build_augmented_messages(
    query: str,
    chat_history: List[Dict],
    confluence_page: str = "",
    search_results: List[Dict] = None,
) -> List[Dict]:
    """
    Build the messages list for the LLM call.
    Includes: system prompt, Confluence page, local doc excerpts, history, and user query.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Source 1: Confluence page
    if confluence_page:
        messages.append({
            "role": "system",
            "content": f"CONFLUENCE PAGE: {ONBOARDING_PAGE_TITLE}\n\n{confluence_page}",
        })

    # Source 2: Local document excerpts from ChromaDB
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

    # Append conversation history (keep last 20 turns to stay within token limits)
    messages.extend(chat_history[-20:])

    # Append current user query
    messages.append({"role": "user", "content": query})

    return messages


# ========================================
# MCP ATLASSIAN TOOLS
# ========================================

def _build_mcp_env() -> dict:
    """Build subprocess env for mcp-atlassian, mapping .env names to what it expects."""
    env = {**os.environ}
    # sooperset/mcp-atlassian Server/DC env var names
    if os.getenv("JIRA_BASE_URL"):
        env["JIRA_URL"] = os.getenv("JIRA_BASE_URL")
    if os.getenv("JIRA_TOKEN"):
        env["JIRA_PERSONAL_TOKEN"] = os.getenv("JIRA_TOKEN")
    if os.getenv("CONFLUENCE_BASE_URL"):
        env["CONFLUENCE_URL"] = os.getenv("CONFLUENCE_BASE_URL")
    if os.getenv("CONFLUENCE_TOKEN"):
        env["CONFLUENCE_PERSONAL_TOKEN"] = os.getenv("CONFLUENCE_TOKEN")
    return env


def _mcp_tools_to_openai(mcp_tools) -> List[Dict]:
    """Convert MCP tool list to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


async def _execute_tool_calls(
    mcp_session: ClientSession,
    tool_calls,
) -> List[Dict]:
    """Execute each tool call via MCP and return a list of tool-role messages."""
    tool_messages = []
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        preview = json.dumps(args, ensure_ascii=False)[:120]
        print(f"  [Tool: {tc.function.name} → {preview}]")
        try:
            result = await mcp_session.call_tool(tc.function.name, args)
            content = "\n".join(
                block.text if hasattr(block, "text") else str(block)
                for block in result.content
            )
        except Exception as exc:
            content = f"Tool error: {exc}"
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": content,
        })
    return tool_messages


# ========================================
# RESPONSE GENERATION
# ========================================

async def generate_response(
    openai_client,
    model: str,
    messages: List[Dict],
    mcp_session: ClientSession,
    openai_tools: List[Dict],
) -> str:
    """Call the LLM with Atlassian tools, looping until no more tool calls remain."""
    msgs = list(messages)  # work on a copy; don't mutate the caller's list

    while True:
        kwargs: Dict[str, Any] = {"model": model, "messages": msgs, "temperature": 0}
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        response = openai_client.chat.completions.create(**kwargs)
        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            return assistant_msg.content or ""

        # Append assistant turn with tool_calls, then execute and append results
        msgs.append({
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_msg.tool_calls
            ],
        })
        tool_results = await _execute_tool_calls(mcp_session, assistant_msg.tool_calls)
        msgs.extend(tool_results)


# ========================================
# INTERACTIVE CHAT LOOP
# ========================================

async def main():
    print("=" * 60)
    print("  RAG Chatbot — Personal Assistant")
    print("=" * 60)

    # 1. Connect to ChromaDB
    collection = get_collection()
    print(f"Connected to ChromaDB — {collection.count()} chunks available")

    # 2. Initialize LLM client
    openai_client, model = create_llm_client()
    print(f"LLM model: {model}")

    # 3. Start mcp-atlassian server and fetch the onboarding page once
    server_params = StdioServerParameters(
        command="mcp-atlassian",
        args=[],
        env=_build_mcp_env(),
    )

    print("Connecting to Atlassian MCP server...", end=" ", flush=True)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            print("OK")

            print(f"Fetching '{ONBOARDING_PAGE_TITLE}'...", end=" ", flush=True)
            try:
                page_result = await mcp_session.call_tool(
                    "confluence_get_page", {"page_id": ONBOARDING_PAGE_ID}
                )
                onboarding_content = "\n".join(
                    block.text if hasattr(block, "text") else str(block)
                    for block in page_result.content
                )
                print(f"OK ({len(onboarding_content)} chars)")
            except Exception as e:
                print(f"FAILED: {e}")
                onboarding_content = ""

            print(f"\nSources:")
            print(f"  1. Confluence: {ONBOARDING_PAGE_TITLE}")
            print(f"  2. ChromaDB: {collection.count()} local document chunks")
            print("\nType your question below. Commands: 'quit' to exit, 'clear' to reset history.\n")

            chat_history: List[Dict] = []

            while True:
                try:
                    query = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not query:
                    continue
                if query.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                if query.lower() == "clear":
                    chat_history.clear()
                    print("-- Chat history cleared --\n")
                    continue

                # Reformulate query for better retrieval
                try:
                    search_query = reformulate_query(openai_client, model, query, chat_history)
                except Exception:
                    search_query = query

                if search_query.lower() != query.lower():
                    print(f"  [Search: {search_query}]")

                # Retrieve relevant local documents from ChromaDB
                search_results = search(collection, search_query)

                # Build prompt with both sources + history
                messages = build_augmented_messages(
                    query, chat_history,
                    confluence_page=onboarding_content,
                    search_results=search_results,
                )

                # Generate LLM response
                try:
                    answer = await generate_response(
                        openai_client, model, messages, mcp_session, openai_tools=[]
                    )
                except Exception as e:
                    print(f"\nError calling LLM: {e}\n")
                    continue

                print(f"\nAssistant: {answer}\n")

                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    asyncio.run(main())
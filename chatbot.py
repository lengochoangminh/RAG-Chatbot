"""
RAG Chatbot — Personal Assistant
=================================
Interactive chatbot that answers questions using documents stored in ChromaDB.
Run ingest_database.py first to populate the vector database, then launch this.
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import chromadb

# ========================================
# CONFIGURATION
# ========================================

# Must match the values in ingest_database.py
COLLECTION_NAME = "knowledge_base"
CHROMA_DB_PATH = "./chroma_db"

SIMILARITY_THRESHOLD = 0.3
TOP_K = 5

load_dotenv()

SYSTEM_PROMPT = """\
You are a helpful personal assistant. Answer the user's question using ONLY \
the document excerpts provided below. Be concise and accurate.

Rules:
- Base your answer strictly on the provided excerpts.
- Reference the source document name and page number when relevant.
- If the excerpts are on the same topic but don't cover the specific detail \
the user asked about, briefly note what IS covered so the user understands \
the scope (e.g. "The policy lists X, Y, Z levels but not internships.").
- If the excerpts are entirely unrelated to the question, simply say the \
available documents do not cover that topic. Do NOT describe unrelated content.
- Do NOT make up information that is not in the excerpts.
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
    search_results: List[Dict],
    chat_history: List[Dict],
) -> List[Dict]:
    """
    Build the messages list for the LLM call.
    Includes: system prompt, retrieved context, conversation history, and user query.
    """
    context_parts = []
    for i, r in enumerate(search_results, 1):
        page = r["metadata"].get("page", "?")
        title = r["metadata"].get("title", "Unknown")
        context_parts.append(
            f"[Source {i}: {title}, Page {page}]\n{r['content']}"
        )

    context_block = "\n\n".join(context_parts) if context_parts else "(No relevant documents found.)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"DOCUMENT EXCERPTS:\n{context_block}"},
    ]

    # Append conversation history (keep last 10 turns to stay within token limits)
    messages.extend(chat_history[-20:])

    # Append current user query
    messages.append({"role": "user", "content": query})

    return messages


# ========================================
# RESPONSE GENERATION
# ========================================

def generate_response(openai_client, model: str, messages: List[Dict]) -> str:
    """Call Azure AI Foundry LLM and return the assistant response."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


# ========================================
# INTERACTIVE CHAT LOOP
# ========================================

def main():
    print("=" * 60)
    print("  RAG Chatbot — Personal Assistant")
    print("=" * 60)

    # 1. Connect to ChromaDB
    collection = get_collection()
    print(f"Connected to ChromaDB — {collection.count()} chunks available")

    # 2. Initialize LLM client
    openai_client, model = create_llm_client()
    print(f"LLM model: {model}")

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

        # Reformulate query: fix typos, resolve follow-ups from history
        try:
            search_query = reformulate_query(openai_client, model, query, chat_history)
        except Exception:
            search_query = query  # fall back to raw input on error

        if search_query.lower() != query.lower():
            print(f"  [Search: {search_query}]")

        # Retrieve relevant documents
        search_results = search(collection, search_query)

        # Build prompt with context + history
        messages = build_augmented_messages(query, search_results, chat_history)

        # Generate LLM response
        try:
            answer = generate_response(openai_client, model, messages)
        except Exception as e:
            print(f"\nError calling LLM: {e}\n")
            continue

        print(f"\nAssistant: {answer}\n")

        # Store turn in history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
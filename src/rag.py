"""ChromaDB-based RAG: build the question bank index and retrieve relevant questions."""
from __future__ import annotations

import json
import os
from typing import Any

from src.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    QUESTIONS_JSON_PATH,
    RAG_TOP_K,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "interview_questions"


def _load_questions() -> list[dict[str, Any]]:
    """Load the question bank from data/questions.json."""
    with open(QUESTIONS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_embedding_function() -> Any:
    """Return a sentence-transformers embedding function for ChromaDB."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_or_create_collection(client: Any) -> Any:
    """Return the ChromaDB collection, creating it if absent."""
    ef = _get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(force_rebuild: bool = False) -> None:
    """
    Build (or rebuild) the ChromaDB vector index from questions.json.

    - On first run, embeds all questions and persists to *CHROMA_PERSIST_DIR*.
    - Subsequent runs skip rebuild unless *force_rebuild* is True.
    """
    import chromadb

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = _get_or_create_collection(client)

    existing_count = collection.count()
    questions = _load_questions()

    if existing_count >= len(questions) and not force_rebuild:
        logger.info(
            "RAG index already contains %d documents; skipping rebuild.", existing_count
        )
        return

    if force_rebuild and existing_count > 0:
        logger.info("Force rebuild: deleting existing collection.")
        client.delete_collection(COLLECTION_NAME)
        collection = _get_or_create_collection(client)

    logger.info("Building RAG index with %d questions…", len(questions))

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for q in questions:
        # Document text = question + key_points joined for richer embedding
        doc_text = q["question"] + " " + " ".join(q.get("key_points", []))
        ids.append(q["id"])
        documents.append(doc_text)
        metadatas.append(
            {
                "role": q["role"],
                "type": q["type"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "key_points": json.dumps(q.get("key_points", [])),
                "source_id": q["id"],
            }
        )

    # Upsert in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info("RAG index built: %d vectors persisted.", collection.count())


def retrieve_questions(
    query: str,
    role: str,
    interview_type: str,
    top_k: int = RAG_TOP_K,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant questions for *role* and *interview_type*.

    Returns a list of dicts with keys: id, question, key_points, difficulty, type.
    Falls back to an empty list (with a log warning) if the index is empty.
    """
    import chromadb

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = _get_or_create_collection(client)

    if collection.count() == 0:
        logger.warning("RAG index is empty. Run build_index() first.")
        return []

    # Build where filter
    if interview_type == "mixed":
        # Mixed: retrieve both technical and behavioral
        where_filter: dict[str, Any] = {"role": {"$eq": role}}
    else:
        where_filter = {
            "$and": [
                {"role": {"$eq": role}},
                {"type": {"$eq": interview_type}},
            ]
        }

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    retrieved: list[dict[str, Any]] = []
    metadatas_list = results.get("metadatas", [[]])[0]
    distances_list = results.get("distances", [[]])[0]

    for meta, dist in zip(metadatas_list, distances_list):
        retrieved.append(
            {
                "id": meta.get("source_id", ""),
                "question": meta.get("question", ""),
                "key_points": json.loads(meta.get("key_points", "[]")),
                "difficulty": meta.get("difficulty", "medium"),
                "type": meta.get("type", interview_type),
                "relevance_score": round(1.0 - float(dist), 4),
            }
        )

    logger.info(
        "Retrieved %d questions for role=%r type=%r", len(retrieved), role, interview_type
    )
    return retrieved


def format_retrieved_for_prompt(retrieved: list[dict[str, Any]]) -> str:
    """Format retrieved questions into a compact string for the prompt."""
    if not retrieved:
        return "(No relevant questions found in the bank; generate general questions.)"

    lines: list[str] = []
    for i, q in enumerate(retrieved, start=1):
        kp = "; ".join(q.get("key_points", [])[:4])  # cap to keep prompt short
        lines.append(
            f"{i}. [ID:{q['id']}] [{q['difficulty']}] {q['question']}\n"
            f"   Key points: {kp}"
        )
    return "\n".join(lines)

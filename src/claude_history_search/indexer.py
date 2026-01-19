"""ChromaDB indexer for conversation embeddings."""

import hashlib
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.config import Settings

from claude_history_search.loader import Conversation, get_default_claude_dir, load_conversations

DEBUG = False

# Default embedding function uses sentence-transformers via ChromaDB
DEFAULT_COLLECTION_NAME = "claude_conversations"

# Batch size for bulk indexing operations (optimized for ChromaDB performance)
DEFAULT_BATCH_SIZE = 100


def get_index_dir() -> Path:
    """Get the directory for storing the ChromaDB index."""
    return get_default_claude_dir() / "search_index"


def get_chroma_client(index_dir: Path | None = None) -> chromadb.Client:
    """Get a ChromaDB persistent client."""
    if index_dir is None:
        index_dir = get_index_dir()

    index_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(index_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.Client) -> chromadb.Collection:
    """Get or create the conversations collection."""
    # Use default embedding function (sentence-transformers all-MiniLM-L6-v2)
    return client.get_or_create_collection(
        name=DEFAULT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def compute_content_hash(content: str) -> str:
    """Compute a hash of the content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def index_conversation(collection: chromadb.Collection, conversation: Conversation) -> bool:
    """Index a single conversation. Returns True if indexed, False if skipped."""
    text_content = conversation.text_content
    if not text_content.strip():
        return False

    # Create document ID from session ID
    doc_id = conversation.session_id

    # Compute content hash for change detection
    content_hash = compute_content_hash(text_content)

    # Check if already indexed with same content
    try:
        existing = collection.get(ids=[doc_id], include=["metadatas"])
        if existing["ids"] and existing["metadatas"]:
            existing_hash = existing["metadatas"][0].get("content_hash", "")
            if existing_hash == content_hash:
                if DEBUG:
                    print(f"Skipping unchanged: {doc_id}")
                return False
    except Exception:
        pass  # Not found, proceed with indexing

    # Prepare metadata
    metadata = {
        "session_id": conversation.session_id,
        "project": conversation.project,
        "summary": conversation.summary[:500],  # Truncate for storage
        "file_path": str(conversation.file_path),
        "content_hash": content_hash,
    }
    if conversation.timestamp:
        metadata["timestamp"] = conversation.timestamp.isoformat()

    # Truncate content for embedding (ChromaDB has limits)
    # Most embedding models have ~512 token context, ~2000 chars is reasonable
    truncated_content = text_content[:8000]

    # Upsert the document
    collection.upsert(
        ids=[doc_id],
        documents=[truncated_content],
        metadatas=[metadata],
    )

    if DEBUG:
        print(f"Indexed: {doc_id}")
    return True


def prepare_conversation_for_batch(
    conversation: Conversation,
) -> tuple[str, str, dict, str] | None:
    """
    Prepare a conversation for batch indexing.

    Returns (doc_id, text_content, metadata, content_hash) or None if empty.
    """
    text_content = conversation.text_content
    if not text_content.strip():
        return None

    doc_id = conversation.session_id
    content_hash = compute_content_hash(text_content)

    metadata = {
        "session_id": conversation.session_id,
        "project": conversation.project,
        "summary": conversation.summary[:500],
        "file_path": str(conversation.file_path),
        "content_hash": content_hash,
    }
    if conversation.timestamp:
        metadata["timestamp"] = conversation.timestamp.isoformat()

    # Truncate content for embedding
    truncated_content = text_content[:8000]

    return (doc_id, truncated_content, metadata, content_hash)


def index_batch(
    collection: chromadb.Collection,
    batch: list[tuple[str, str, dict, str]],
) -> tuple[int, int]:
    """
    Index a batch of conversations.

    Returns (indexed_count, skipped_count).
    """
    if not batch:
        return 0, 0

    doc_ids = [item[0] for item in batch]

    # Check which documents already exist with same content hash
    try:
        existing = collection.get(ids=doc_ids, include=["metadatas"])
        existing_map = {}
        if existing["ids"] and existing["metadatas"]:
            for i, doc_id in enumerate(existing["ids"]):
                existing_map[doc_id] = existing["metadatas"][i].get("content_hash", "")
    except Exception:
        existing_map = {}

    # Filter to only documents that need updating
    to_upsert_ids = []
    to_upsert_docs = []
    to_upsert_meta = []
    skipped = 0

    for doc_id, content, metadata, content_hash in batch:
        if doc_id in existing_map and existing_map[doc_id] == content_hash:
            skipped += 1
            if DEBUG:
                print(f"Skipping unchanged: {doc_id}")
        else:
            to_upsert_ids.append(doc_id)
            to_upsert_docs.append(content)
            to_upsert_meta.append(metadata)

    # Batch upsert
    if to_upsert_ids:
        collection.upsert(
            ids=to_upsert_ids,
            documents=to_upsert_docs,
            metadatas=to_upsert_meta,
        )
        if DEBUG:
            print(f"Batch indexed {len(to_upsert_ids)} documents")

    return len(to_upsert_ids), skipped


def build_index(
    claude_dir: Path | None = None,
    index_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict:
    """
    Build or update the search index. Returns statistics.

    Supports 1000+ conversation files through efficient batch processing.

    Args:
        claude_dir: Path to Claude directory (default: ~/.claude)
        index_dir: Path to index directory (default: ~/.claude/search_index)
        batch_size: Number of conversations to process per batch (default: 100)
        progress_callback: Optional callback(total_processed, indexed, skipped)
            called after each batch for progress tracking

    Returns:
        dict with stats: total_files, indexed, skipped, errors
    """
    client = get_chroma_client(index_dir)
    collection = get_or_create_collection(client)

    stats = {
        "total_files": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
    }

    batch: list[tuple[str, str, dict, str]] = []

    for conversation in load_conversations(claude_dir):
        stats["total_files"] += 1
        try:
            prepared = prepare_conversation_for_batch(conversation)
            if prepared is None:
                stats["skipped"] += 1
                continue

            batch.append(prepared)

            # Process batch when full
            if len(batch) >= batch_size:
                indexed, skipped = index_batch(collection, batch)
                stats["indexed"] += indexed
                stats["skipped"] += skipped
                batch = []

                # Progress callback
                if progress_callback:
                    progress_callback(
                        stats["total_files"],
                        stats["indexed"],
                        stats["skipped"],
                    )

        except Exception as e:
            stats["errors"] += 1
            if DEBUG:
                print(f"Error indexing {conversation.session_id}: {e}")

    # Process remaining batch
    if batch:
        try:
            indexed, skipped = index_batch(collection, batch)
            stats["indexed"] += indexed
            stats["skipped"] += skipped
        except Exception as e:
            stats["errors"] += len(batch)
            if DEBUG:
                print(f"Error indexing final batch: {e}")

    # Final progress callback
    if progress_callback:
        progress_callback(
            stats["total_files"],
            stats["indexed"],
            stats["skipped"],
        )

    return stats


def get_index_stats(index_dir: Path | None = None) -> dict:
    """Get statistics about the current index."""
    try:
        client = get_chroma_client(index_dir)
        collection = get_or_create_collection(client)
        count = collection.count()
        return {
            "indexed_conversations": count,
            "index_path": str(get_index_dir() if index_dir is None else index_dir),
        }
    except Exception as e:
        return {
            "error": str(e),
            "indexed_conversations": 0,
        }


def clear_index(index_dir: Path | None = None) -> None:
    """Clear the search index."""
    client = get_chroma_client(index_dir)
    try:
        client.delete_collection(DEFAULT_COLLECTION_NAME)
    except Exception:
        pass  # Collection might not exist

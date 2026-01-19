"""Tests for the indexer module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_history_search.indexer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    build_index,
    clear_index,
    compute_content_hash,
    get_chroma_client,
    get_index_dir,
    get_index_stats,
    get_or_create_collection,
    index_batch,
    index_conversation,
    prepare_conversation_for_batch,
)
from claude_history_search.loader import Conversation


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for test index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_conversation():
    """Create a sample Conversation object for testing."""
    return Conversation(
        session_id="test-session-123",
        project="test-project",
        summaries=["This is a test summary about Python debugging"],
        messages=[
            {"role": "user", "content": "Help me debug Python"},
            {"role": "assistant", "content": [{"type": "text", "text": "I'll help you."}]},
        ],
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        file_path=Path("/test/path/test-session-123.jsonl"),
    )


@pytest.fixture
def empty_conversation():
    """Create a conversation with no text content."""
    return Conversation(
        session_id="empty-session",
        project="test-project",
        summaries=[],
        messages=[],
        timestamp=None,
        file_path=Path("/test/path/empty.jsonl"),
    )


class TestGetIndexDir:
    """Tests for get_index_dir function."""

    def test_returns_correct_path(self):
        """Test that get_index_dir returns ~/.claude/search_index."""
        expected = Path.home() / ".claude" / "search_index"
        assert get_index_dir() == expected


class TestGetChromaClient:
    """Tests for get_chroma_client function."""

    def test_creates_directory_if_not_exists(self, temp_index_dir):
        """Test that client creation creates the index directory."""
        index_path = temp_index_dir / "new_subdir"
        assert not index_path.exists()

        client = get_chroma_client(index_path)

        assert index_path.exists()
        assert client is not None

    def test_uses_default_path_when_none(self):
        """Test that None index_dir uses default path."""
        # This test verifies behavior, actual creation is expensive
        with patch("claude_history_search.indexer.get_index_dir") as mock_get_dir:
            mock_path = Path("/mock/path")
            mock_get_dir.return_value = mock_path
            # Would fail to create actual client but tests the path logic
            try:
                get_chroma_client(None)
            except Exception:
                pass  # Expected since mock path doesn't exist
            mock_get_dir.assert_called_once()

    def test_returns_persistent_client(self, temp_index_dir):
        """Test that returned client is a persistent client."""
        client = get_chroma_client(temp_index_dir)
        # PersistentClient has a _persist_directory attribute
        assert hasattr(client, "_identifier")


class TestGetOrCreateCollection:
    """Tests for get_or_create_collection function."""

    def test_creates_collection_with_correct_name(self, temp_index_dir):
        """Test that collection is created with the correct name."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        assert collection.name == DEFAULT_COLLECTION_NAME

    def test_returns_same_collection_on_repeated_calls(self, temp_index_dir):
        """Test that repeated calls return the same collection."""
        client = get_chroma_client(temp_index_dir)

        collection1 = get_or_create_collection(client)
        collection2 = get_or_create_collection(client)

        assert collection1.name == collection2.name


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_returns_16_char_string(self):
        """Test that hash is truncated to 16 characters."""
        content_hash = compute_content_hash("test content")
        assert len(content_hash) == 16

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        content = "This is some test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        hash1 = compute_content_hash("content one")
        hash2 = compute_content_hash("content two")
        assert hash1 != hash2

    def test_empty_string_produces_hash(self):
        """Test that empty string still produces a hash."""
        content_hash = compute_content_hash("")
        assert len(content_hash) == 16


class TestIndexConversation:
    """Tests for index_conversation function."""

    def test_indexes_valid_conversation(self, temp_index_dir, sample_conversation):
        """Test that a valid conversation is indexed."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        result = index_conversation(collection, sample_conversation)

        assert result is True
        assert collection.count() == 1

    def test_skips_empty_conversation(self, temp_index_dir, empty_conversation):
        """Test that empty conversation is not indexed."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        result = index_conversation(collection, empty_conversation)

        assert result is False
        assert collection.count() == 0

    def test_skips_unchanged_conversation(self, temp_index_dir, sample_conversation):
        """Test that unchanged conversation is skipped on re-index."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        # Index first time
        result1 = index_conversation(collection, sample_conversation)
        assert result1 is True

        # Index again - should skip
        result2 = index_conversation(collection, sample_conversation)
        assert result2 is False

    def test_updates_changed_conversation(self, temp_index_dir, sample_conversation):
        """Test that changed conversation is updated."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        # Index first time
        index_conversation(collection, sample_conversation)

        # Modify conversation content
        sample_conversation.summaries = ["Modified summary content"]

        # Index again - should update
        result = index_conversation(collection, sample_conversation)
        assert result is True

    def test_metadata_includes_required_fields(self, temp_index_dir, sample_conversation):
        """Test that indexed metadata includes all required fields."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        index_conversation(collection, sample_conversation)

        # Get the indexed document
        results = collection.get(ids=[sample_conversation.session_id], include=["metadatas"])

        metadata = results["metadatas"][0]
        assert "session_id" in metadata
        assert "project" in metadata
        assert "summary" in metadata
        assert "file_path" in metadata
        assert "content_hash" in metadata
        assert "timestamp" in metadata

    def test_truncates_long_summary(self, temp_index_dir):
        """Test that summary is truncated to 500 chars."""
        long_summary = "A" * 600
        conversation = Conversation(
            session_id="long-summary-test",
            project="test",
            summaries=[long_summary],
            messages=[{"role": "user", "content": "test"}],
            timestamp=None,
            file_path=Path("/test"),
        )

        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)
        index_conversation(collection, conversation)

        results = collection.get(ids=["long-summary-test"], include=["metadatas"])
        stored_summary = results["metadatas"][0]["summary"]

        assert len(stored_summary) == 500


class TestPrepareConversationForBatch:
    """Tests for prepare_conversation_for_batch function."""

    def test_returns_tuple_for_valid_conversation(self, sample_conversation):
        """Test that valid conversation returns a tuple."""
        result = prepare_conversation_for_batch(sample_conversation)

        assert result is not None
        assert len(result) == 4
        doc_id, content, metadata, content_hash = result
        assert doc_id == sample_conversation.session_id
        assert len(content) > 0
        assert "project" in metadata
        assert len(content_hash) == 16

    def test_returns_none_for_empty_conversation(self, empty_conversation):
        """Test that empty conversation returns None."""
        result = prepare_conversation_for_batch(empty_conversation)
        assert result is None

    def test_truncates_content_to_8000_chars(self):
        """Test that content is truncated to 8000 characters."""
        long_content = "A" * 10000
        conversation = Conversation(
            session_id="long-content",
            project="test",
            summaries=[long_content],
            messages=[],
            timestamp=None,
            file_path=Path("/test"),
        )

        result = prepare_conversation_for_batch(conversation)
        _, content, _, _ = result

        assert len(content) == 8000

    def test_metadata_includes_timestamp_when_available(self, sample_conversation):
        """Test that metadata includes timestamp when conversation has one."""
        result = prepare_conversation_for_batch(sample_conversation)
        _, _, metadata, _ = result

        assert "timestamp" in metadata
        assert sample_conversation.timestamp.isoformat() in metadata["timestamp"]


class TestIndexBatch:
    """Tests for index_batch function."""

    def test_indexes_batch_of_conversations(self, temp_index_dir):
        """Test that batch indexing works correctly."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        batch = [
            ("id1", "content 1", {"project": "p1", "summary": "s1"}, "hash1"),
            ("id2", "content 2", {"project": "p2", "summary": "s2"}, "hash2"),
            ("id3", "content 3", {"project": "p3", "summary": "s3"}, "hash3"),
        ]

        indexed, skipped = index_batch(collection, batch)

        assert indexed == 3
        assert skipped == 0
        assert collection.count() == 3

    def test_skips_unchanged_documents(self, temp_index_dir):
        """Test that unchanged documents are skipped in batch."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        batch = [
            ("id1", "content 1", {"project": "p1", "summary": "s1", "content_hash": "hash1"}, "hash1"),
        ]

        # First batch
        index_batch(collection, batch)

        # Second batch with same hash
        indexed, skipped = index_batch(collection, batch)

        assert indexed == 0
        assert skipped == 1

    def test_empty_batch_returns_zeros(self, temp_index_dir):
        """Test that empty batch returns (0, 0)."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        indexed, skipped = index_batch(collection, [])

        assert indexed == 0
        assert skipped == 0


class TestBuildIndex:
    """Tests for build_index function."""

    @pytest.fixture
    def sample_claude_dir(self, tmp_path):
        """Create a sample Claude directory structure with conversations."""
        claude_dir = tmp_path / ".claude"
        projects_dir = claude_dir / "projects" / "test-project"
        projects_dir.mkdir(parents=True)

        # Create sample conversation files
        for i in range(3):
            file_path = projects_dir / f"session-{i:03d}.jsonl"
            with open(file_path, "w") as f:
                f.write(json.dumps({"type": "summary", "summary": f"Conversation {i}"}) + "\n")
                f.write(json.dumps({
                    "type": "user",
                    "message": {"role": "user", "content": f"Message {i}"},
                    "timestamp": "2024-01-15T10:00:00Z",
                }) + "\n")

        return claude_dir

    def test_builds_index_successfully(self, sample_claude_dir, temp_index_dir):
        """Test that build_index processes all conversations."""
        stats = build_index(claude_dir=sample_claude_dir, index_dir=temp_index_dir)

        assert stats["total_files"] == 3
        assert stats["indexed"] == 3
        assert stats["errors"] == 0

    def test_returns_correct_stats(self, sample_claude_dir, temp_index_dir):
        """Test that stats dictionary has all required keys."""
        stats = build_index(claude_dir=sample_claude_dir, index_dir=temp_index_dir)

        assert "total_files" in stats
        assert "indexed" in stats
        assert "skipped" in stats
        assert "errors" in stats

    def test_incremental_indexing_skips_unchanged(self, sample_claude_dir, temp_index_dir):
        """Test that re-indexing skips unchanged files."""
        # First index
        stats1 = build_index(claude_dir=sample_claude_dir, index_dir=temp_index_dir)
        assert stats1["indexed"] == 3

        # Second index - should skip all
        stats2 = build_index(claude_dir=sample_claude_dir, index_dir=temp_index_dir)
        assert stats2["indexed"] == 0
        assert stats2["skipped"] == 3

    def test_progress_callback_is_called(self, sample_claude_dir, temp_index_dir):
        """Test that progress callback is invoked during indexing."""
        callback_calls = []

        def progress_callback(total, indexed, skipped):
            callback_calls.append((total, indexed, skipped))

        build_index(
            claude_dir=sample_claude_dir,
            index_dir=temp_index_dir,
            progress_callback=progress_callback,
        )

        assert len(callback_calls) > 0

    def test_custom_batch_size(self, sample_claude_dir, temp_index_dir):
        """Test that custom batch size is respected."""
        stats = build_index(
            claude_dir=sample_claude_dir,
            index_dir=temp_index_dir,
            batch_size=1,  # Process one at a time
        )

        # Should still process all files
        assert stats["total_files"] == 3

    def test_handles_empty_directory(self, tmp_path, temp_index_dir):
        """Test handling of empty Claude directory."""
        empty_claude_dir = tmp_path / "empty_claude"
        empty_claude_dir.mkdir()

        stats = build_index(claude_dir=empty_claude_dir, index_dir=temp_index_dir)

        assert stats["total_files"] == 0
        assert stats["indexed"] == 0


class TestGetIndexStats:
    """Tests for get_index_stats function."""

    def test_returns_count_for_empty_index(self, temp_index_dir):
        """Test stats for empty index."""
        stats = get_index_stats(index_dir=temp_index_dir)

        assert stats["indexed_conversations"] == 0
        assert "index_path" in stats

    def test_returns_correct_count_after_indexing(self, temp_index_dir):
        """Test that stats reflect indexed documents."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        # Add some documents
        collection.upsert(
            ids=["doc1", "doc2"],
            documents=["content 1", "content 2"],
            metadatas=[{"key": "val1"}, {"key": "val2"}],
        )

        stats = get_index_stats(index_dir=temp_index_dir)

        assert stats["indexed_conversations"] == 2

    def test_handles_error_gracefully(self):
        """Test that errors are handled and returned in stats."""
        with patch("claude_history_search.indexer.get_chroma_client") as mock_client:
            mock_client.side_effect = Exception("Connection error")

            stats = get_index_stats(index_dir=Path("/nonexistent"))

            assert "error" in stats
            assert stats["indexed_conversations"] == 0


class TestClearIndex:
    """Tests for clear_index function."""

    def test_clears_existing_index(self, temp_index_dir):
        """Test that clear_index removes all indexed data."""
        client = get_chroma_client(temp_index_dir)
        collection = get_or_create_collection(client)

        # Add some documents
        collection.upsert(
            ids=["doc1"],
            documents=["content"],
            metadatas=[{"key": "val"}],
        )
        assert collection.count() == 1

        # Clear the index
        clear_index(index_dir=temp_index_dir)

        # Verify cleared (collection should be recreated empty)
        stats = get_index_stats(index_dir=temp_index_dir)
        assert stats["indexed_conversations"] == 0

    def test_handles_nonexistent_collection(self, temp_index_dir):
        """Test that clearing nonexistent collection doesn't raise error."""
        # This should not raise an exception
        clear_index(index_dir=temp_index_dir)

        stats = get_index_stats(index_dir=temp_index_dir)
        assert stats["indexed_conversations"] == 0


class TestConstants:
    """Tests for module constants."""

    def test_default_collection_name(self):
        """Test that default collection name is set."""
        assert DEFAULT_COLLECTION_NAME == "claude_conversations"

    def test_default_batch_size(self):
        """Test that default batch size is reasonable."""
        assert DEFAULT_BATCH_SIZE == 100
        assert DEFAULT_BATCH_SIZE > 0

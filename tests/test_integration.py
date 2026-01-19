"""Integration tests for Claude History Search.

These tests verify that all components work together correctly.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from claude_history_search.indexer import (
    build_index,
    clear_index,
    get_chroma_client,
    get_index_stats,
    get_or_create_collection,
)
from claude_history_search.loader import (
    Conversation,
    find_conversation_files,
    load_conversations,
    parse_conversation_file,
)
from claude_history_search.search import (
    SearchResponse,
    SearchResult,
    search_by_topic,
    search_conversations,
)


@pytest.fixture
def full_test_environment(tmp_path):
    """Create a complete test environment with conversation files."""
    claude_dir = tmp_path / ".claude"
    index_dir = tmp_path / "search_index"

    # Create multiple projects with various conversations
    projects = {
        "web-app": [
            {
                "id": "session-web-001",
                "summary": "Setting up React frontend with TypeScript",
                "messages": [
                    "How do I set up a React project with TypeScript?",
                    "I'll help you set up React with TypeScript using Vite...",
                ],
            },
            {
                "id": "session-web-002",
                "summary": "Debugging CSS grid layout issues",
                "messages": [
                    "My CSS grid is not displaying correctly",
                    "Let me help you debug the CSS grid layout...",
                ],
            },
        ],
        "backend-api": [
            {
                "id": "session-api-001",
                "summary": "Building REST API with FastAPI",
                "messages": [
                    "Help me build a REST API with FastAPI",
                    "I'll guide you through building a FastAPI application...",
                ],
            },
            {
                "id": "session-api-002",
                "summary": "Database connection pooling optimization",
                "messages": [
                    "How do I optimize PostgreSQL connection pooling?",
                    "Let's optimize your database connections...",
                ],
            },
        ],
        "ml-project": [
            {
                "id": "session-ml-001",
                "summary": "Training a sentiment analysis model",
                "messages": [
                    "How do I train a sentiment analysis model?",
                    "Let's build a sentiment classifier using transformers...",
                ],
            },
        ],
    }

    for project_name, conversations in projects.items():
        project_dir = claude_dir / "projects" / project_name
        project_dir.mkdir(parents=True)

        for conv in conversations:
            file_path = project_dir / f"{conv['id']}.jsonl"
            with open(file_path, "w") as f:
                # Write summary
                f.write(json.dumps({"type": "summary", "summary": conv["summary"]}) + "\n")
                # Write messages
                for i, msg in enumerate(conv["messages"]):
                    role = "user" if i % 2 == 0 else "assistant"
                    if role == "user":
                        f.write(json.dumps({
                            "type": "user",
                            "message": {"role": "user", "content": msg},
                            "timestamp": "2024-01-15T10:00:00Z",
                        }) + "\n")
                    else:
                        f.write(json.dumps({
                            "type": "assistant",
                            "message": {"role": "assistant", "content": [{"type": "text", "text": msg}]},
                        }) + "\n")

    return {
        "claude_dir": claude_dir,
        "index_dir": index_dir,
        "projects": projects,
    }


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_index_and_search_workflow(self, full_test_environment):
        """Test the complete workflow: load -> index -> search."""
        env = full_test_environment

        # Step 1: Find and load conversations
        files = list(find_conversation_files(env["claude_dir"]))
        assert len(files) == 5  # 2 + 2 + 1 conversations

        conversations = list(load_conversations(env["claude_dir"]))
        assert len(conversations) == 5

        # Step 2: Build index
        stats = build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        assert stats["total_files"] == 5
        assert stats["indexed"] == 5
        assert stats["errors"] == 0

        # Step 3: Verify index stats
        index_stats = get_index_stats(index_dir=env["index_dir"])
        assert index_stats["indexed_conversations"] == 5

        # Step 4: Search for specific content
        response = search_conversations(
            "React TypeScript frontend",
            index_dir=env["index_dir"],
        )
        assert len(response.results) > 0
        # Web app conversation should be highly relevant
        top_result = response.results[0]
        assert "web" in top_result.project.lower() or "react" in top_result.summary.lower()

    def test_search_with_filters(self, full_test_environment):
        """Test search with various filter combinations."""
        env = full_test_environment

        # Build index first
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        # Search with project filter
        response = search_conversations(
            "database",
            index_dir=env["index_dir"],
            project_filter="backend-api",
        )
        for result in response.results:
            assert "backend-api" in result.project.lower()

        # Search with non-matching filter
        response = search_conversations(
            "database",
            index_dir=env["index_dir"],
            project_filter="nonexistent-project",
        )
        assert len(response.results) == 0

    def test_topic_search_workflow(self, full_test_environment):
        """Test topic-based search."""
        env = full_test_environment

        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        # Search by API topic
        response = search_by_topic("api", index_dir=env["index_dir"])
        assert len(response.results) > 0

        # Search by database topic
        response = search_by_topic("database", index_dir=env["index_dir"])
        assert len(response.results) > 0


class TestIncrementalIndexing:
    """Test incremental indexing behavior."""

    def test_reindexing_skips_unchanged_files(self, full_test_environment):
        """Test that unchanged files are skipped on re-index."""
        env = full_test_environment

        # First index
        stats1 = build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        assert stats1["indexed"] == 5

        # Second index - all should be skipped
        stats2 = build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        assert stats2["indexed"] == 0
        assert stats2["skipped"] == 5

    def test_new_files_are_indexed(self, full_test_environment):
        """Test that new files are picked up on re-index."""
        env = full_test_environment

        # First index
        stats1 = build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        assert stats1["indexed"] == 5

        # Add new conversation file
        new_project_dir = env["claude_dir"] / "projects" / "new-project"
        new_project_dir.mkdir(parents=True)
        new_file = new_project_dir / "session-new.jsonl"
        with open(new_file, "w") as f:
            f.write(json.dumps({"type": "summary", "summary": "New conversation"}) + "\n")
            f.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "Hello"},
            }) + "\n")

        # Re-index
        stats2 = build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        assert stats2["indexed"] == 1  # Only the new file
        assert stats2["skipped"] == 5  # Previous files skipped


class TestSearchQuality:
    """Test search result quality and relevance."""

    def test_relevant_results_rank_higher(self, full_test_environment):
        """Test that more relevant results have higher scores."""
        env = full_test_environment
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        response = search_conversations(
            "FastAPI REST API endpoint",
            index_dir=env["index_dir"],
        )
        assert len(response.results) >= 2

        # Results should be sorted by relevance
        for i in range(len(response.results) - 1):
            assert response.results[i].relevance_score >= response.results[i + 1].relevance_score

    def test_semantic_search_finds_related_content(self, full_test_environment):
        """Test that semantic search finds conceptually related content."""
        env = full_test_environment
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        # Search for "SQL queries" should find database-related content
        response = search_conversations(
            "SQL queries optimization",
            index_dir=env["index_dir"],
        )
        assert len(response.results) > 0

        # Database optimization conversation should be found
        found_db_content = any(
            "database" in r.summary.lower() or "postgresql" in r.summary.lower()
            for r in response.results
        )
        assert found_db_content, "Expected to find database-related content"


class TestClearAndRebuild:
    """Test index clearing and rebuilding."""

    def test_clear_and_rebuild_cycle(self, full_test_environment):
        """Test clearing index and rebuilding from scratch."""
        env = full_test_environment

        # Build initial index
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        stats1 = get_index_stats(index_dir=env["index_dir"])
        assert stats1["indexed_conversations"] == 5

        # Clear index
        clear_index(index_dir=env["index_dir"])
        stats2 = get_index_stats(index_dir=env["index_dir"])
        assert stats2["indexed_conversations"] == 0

        # Rebuild
        build_stats = build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])
        assert build_stats["indexed"] == 5  # All re-indexed since cleared

        stats3 = get_index_stats(index_dir=env["index_dir"])
        assert stats3["indexed_conversations"] == 5


class TestDataIntegrity:
    """Test data integrity through the pipeline."""

    def test_metadata_preserved_through_indexing(self, full_test_environment):
        """Test that conversation metadata is preserved."""
        env = full_test_environment
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        # Get indexed document directly
        client = get_chroma_client(env["index_dir"])
        collection = get_or_create_collection(client)

        results = collection.get(ids=["session-web-001"], include=["metadatas"])

        assert len(results["ids"]) == 1
        metadata = results["metadatas"][0]

        assert metadata["session_id"] == "session-web-001"
        assert metadata["project"] == "web-app"
        assert "React" in metadata["summary"]  # Summary preserved
        assert "file_path" in metadata

    def test_search_results_contain_correct_data(self, full_test_environment):
        """Test that search results contain correct metadata."""
        env = full_test_environment
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        response = search_conversations(
            "sentiment analysis",
            index_dir=env["index_dir"],
        )
        assert len(response.results) > 0

        # Find the ML conversation
        ml_result = next(
            (r for r in response.results if r.session_id == "session-ml-001"),
            None
        )
        assert ml_result is not None
        assert ml_result.project == "ml-project"
        assert "sentiment" in ml_result.summary.lower()


class TestErrorHandling:
    """Test error handling in the integration."""

    def test_handles_corrupted_files_gracefully(self, tmp_path):
        """Test that corrupted files don't break the entire index."""
        claude_dir = tmp_path / ".claude"
        index_dir = tmp_path / "index"

        project_dir = claude_dir / "projects" / "test"
        project_dir.mkdir(parents=True)

        # Create valid file
        valid_file = project_dir / "valid.jsonl"
        with open(valid_file, "w") as f:
            f.write(json.dumps({"type": "summary", "summary": "Valid file"}) + "\n")
            f.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "Hello"},
            }) + "\n")

        # Create corrupted file
        corrupt_file = project_dir / "corrupt.jsonl"
        with open(corrupt_file, "w") as f:
            f.write("not valid json{{{{\n")
            f.write("more garbage\n")

        # Create empty file
        empty_file = project_dir / "empty.jsonl"
        empty_file.touch()

        # Should handle gracefully
        stats = build_index(claude_dir=claude_dir, index_dir=index_dir)

        # Valid file should be indexed
        assert stats["indexed"] >= 1
        # No crashes
        assert stats["errors"] == 0  # Parsing errors are handled internally

    def test_search_on_empty_index(self, tmp_path):
        """Test searching an empty index returns gracefully."""
        index_dir = tmp_path / "empty_index"

        response = search_conversations("test query", index_dir=index_dir)

        assert isinstance(response, SearchResponse)
        assert response.results == []
        assert response.query == "test query"
        assert response.elapsed_ms >= 0


class TestPerformance:
    """Performance-related integration tests."""

    def test_search_returns_quickly(self, full_test_environment):
        """Test that search returns within acceptable time."""
        env = full_test_environment
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        response = search_conversations(
            "programming question",
            index_dir=env["index_dir"],
        )

        # Search should complete in under 2 seconds
        assert response.elapsed_seconds < 2.0

    def test_multiple_searches_are_consistent(self, full_test_environment):
        """Test that repeated searches return consistent results."""
        env = full_test_environment
        build_index(claude_dir=env["claude_dir"], index_dir=env["index_dir"])

        query = "React frontend development"

        response1 = search_conversations(query, index_dir=env["index_dir"])
        response2 = search_conversations(query, index_dir=env["index_dir"])

        # Same query should return same results
        assert len(response1.results) == len(response2.results)
        for r1, r2 in zip(response1.results, response2.results):
            assert r1.session_id == r2.session_id

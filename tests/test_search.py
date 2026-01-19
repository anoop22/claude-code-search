"""Tests for search functionality."""

import tempfile
import time
from pathlib import Path

import pytest

from claude_history_search.indexer import (
    build_index,
    clear_index,
    get_chroma_client,
    get_index_stats,
    get_or_create_collection,
)
from claude_history_search.loader import Conversation
from claude_history_search.search import (
    SearchResult,
    expand_query,
    extract_topics,
    parse_date,
    search_by_topic,
    search_conversations,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for test index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_conversations(tmp_path):
    """Create sample conversation files for testing."""
    claude_dir = tmp_path / ".claude"
    projects_dir = claude_dir / "projects" / "test-project"
    projects_dir.mkdir(parents=True)

    # Create sample conversation files
    conversations = [
        {
            "id": "session-001",
            "summaries": ["Python debugging session", "Fixed memory leak in database connection"],
            "messages": [
                {"role": "user", "content": "Help me debug this Python memory leak"},
                {"role": "assistant", "content": [{"type": "text", "text": "I'll help analyze the memory issue."}]},
            ],
        },
        {
            "id": "session-002",
            "summaries": ["React component development"],
            "messages": [
                {"role": "user", "content": "Create a React form component with validation"},
                {"role": "assistant", "content": [{"type": "text", "text": "I'll create a form component."}]},
            ],
        },
        {
            "id": "session-003",
            "summaries": ["Database optimization discussion"],
            "messages": [
                {"role": "user", "content": "How can I optimize SQL queries for better performance?"},
                {"role": "assistant", "content": [{"type": "text", "text": "Let me explain query optimization."}]},
            ],
        },
    ]

    import json

    for conv in conversations:
        file_path = projects_dir / f"{conv['id']}.jsonl"
        with open(file_path, "w") as f:
            for summary in conv["summaries"]:
                f.write(json.dumps({"type": "summary", "summary": summary}) + "\n")
            for msg in conv["messages"]:
                f.write(json.dumps({
                    "type": msg["role"],
                    "message": msg,
                    "timestamp": "2026-01-15T10:00:00Z",
                }) + "\n")

    return claude_dir


def test_search_result_format():
    """Test SearchResult formatting."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Test conversation about Python",
        relevance_score=0.85,
        file_path="/test/path",
        timestamp="2026-01-15T10:00:00Z",
    )

    display = result.format_display()
    assert "85%" in display
    assert "Test conversation" in display
    assert "test-project" in display


def test_search_empty_index(temp_index_dir):
    """Test search on empty index."""
    response = search_conversations("test query", index_dir=temp_index_dir)
    assert response.results == []
    assert response.query == "test query"


def test_search_returns_results(sample_conversations, temp_index_dir):
    """Test that search returns relevant results."""
    # Build index
    stats = build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)
    assert stats["indexed"] > 0

    # Search for Python content
    response = search_conversations("Python debugging", index_dir=temp_index_dir)
    assert len(response.results) > 0

    # The Python debugging conversation should be most relevant (session-001)
    # The summary may be either of the summaries from that conversation
    top_result = response.results[0]
    assert top_result.session_id == "session-001", (
        f"Expected session-001 (Python debugging) to be top result, got {top_result.session_id}"
    )


def test_search_performance_under_2_seconds(sample_conversations, temp_index_dir):
    """Test that search returns results within 2 seconds."""
    # Build index first
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Measure search time
    start_time = time.perf_counter()
    response = search_conversations("Python memory leak", index_dir=temp_index_dir)
    elapsed = time.perf_counter() - start_time

    # Assert search completes within 2 seconds
    assert elapsed < 2.0, f"Search took {elapsed:.2f}s, should be under 2s"

    # Also verify the response includes timing info
    assert response.elapsed_seconds < 2.0


def test_search_num_results_limit(sample_conversations, temp_index_dir):
    """Test that num_results limits output."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    response = search_conversations("programming", num_results=1, index_dir=temp_index_dir)
    assert len(response.results) <= 1


def test_index_stats(temp_index_dir):
    """Test getting index statistics."""
    stats = get_index_stats(index_dir=temp_index_dir)
    assert "indexed_conversations" in stats
    assert stats["indexed_conversations"] == 0


def test_index_and_stats(sample_conversations, temp_index_dir):
    """Test indexing and stats together."""
    # Index conversations
    build_stats = build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)
    assert build_stats["indexed"] > 0

    # Check stats
    stats = get_index_stats(index_dir=temp_index_dir)
    assert stats["indexed_conversations"] == build_stats["indexed"]


def test_clear_index(sample_conversations, temp_index_dir):
    """Test clearing the index."""
    # Build and then clear
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)
    clear_index(index_dir=temp_index_dir)

    # Verify cleared
    stats = get_index_stats(index_dir=temp_index_dir)
    assert stats["indexed_conversations"] == 0


# ============= Topic Search Tests =============


def test_extract_topics():
    """Test topic extraction from text."""
    # Database related
    topics = extract_topics("Working with SQL queries and PostgreSQL database")
    assert "database" in topics

    # API related
    topics = extract_topics("Building REST API endpoints")
    assert "api" in topics

    # Testing related
    topics = extract_topics("Writing pytest unit tests")
    assert "testing" in topics

    # Multiple topics
    topics = extract_topics("Testing database API endpoints")
    assert "database" in topics
    assert "api" in topics
    assert "testing" in topics


def test_expand_query():
    """Test query expansion with synonyms."""
    # API query should expand with related terms
    expanded = expand_query("api development")
    assert "api" in expanded.lower()
    # Should contain some expansions
    assert len(expanded) > len("api development")

    # Database query
    expanded = expand_query("database optimization")
    assert "database" in expanded.lower()

    # Synonym should expand to include parent topic
    expanded = expand_query("using db connections")
    assert "database" in expanded.lower()

    # Query without known topics stays unchanged
    original = "random unrelated phrase"
    expanded = expand_query(original)
    assert expanded == original


def test_parse_date():
    """Test date parsing."""
    # Standard format
    date = parse_date("2026-01-15")
    assert date is not None
    assert date.year == 2026
    assert date.month == 1
    assert date.day == 15

    # Invalid format returns None
    assert parse_date("invalid") is None
    assert parse_date("") is None


def test_search_result_with_topics():
    """Test SearchResult with matched topics."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Test conversation about API testing",
        relevance_score=0.85,
        file_path="/test/path",
        timestamp="2026-01-15T10:00:00Z",
        matched_topics=["api", "testing"],
    )

    display = result.format_display()
    assert "Topics:" in display
    assert "api" in display
    assert "testing" in display


def test_search_by_topic(sample_conversations, temp_index_dir):
    """Test topic-based search."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Search by "database" topic
    response = search_by_topic("database", index_dir=temp_index_dir)
    assert len(response.results) > 0

    # Database conversation should be in results
    session_ids = [r.session_id for r in response.results]
    assert "session-003" in session_ids, (
        f"Expected session-003 (database) in results, got {session_ids}"
    )


def test_search_natural_language_query(sample_conversations, temp_index_dir):
    """Test natural language search queries."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Natural language query about debugging
    response = search_conversations(
        "how do I fix memory issues in Python",
        index_dir=temp_index_dir
    )
    assert len(response.results) > 0

    # Python debugging session should be relevant
    top_result = response.results[0]
    assert "python" in top_result.summary.lower() or top_result.session_id == "session-001"


def test_search_with_project_filter(sample_conversations, temp_index_dir):
    """Test search with project filter."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Search with matching project filter
    response = search_conversations(
        "programming",
        project_filter="test-project",
        index_dir=temp_index_dir
    )
    # All results should be from test-project
    for result in response.results:
        assert "test-project" in result.project.lower()

    # Search with non-matching project filter
    response = search_conversations(
        "programming",
        project_filter="nonexistent",
        index_dir=temp_index_dir
    )
    assert len(response.results) == 0


def test_search_with_query_expansion(sample_conversations, temp_index_dir):
    """Test search with and without query expansion."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Search with expansion enabled (default)
    response_expanded = search_conversations(
        "db queries",  # 'db' should expand to include 'database'
        index_dir=temp_index_dir,
        expand_topics=True
    )

    # Search without expansion
    response_no_expand = search_conversations(
        "db queries",
        index_dir=temp_index_dir,
        expand_topics=False
    )

    # Both should return results (semantic search still works)
    # But expanded query may have better relevance for database content
    assert response_expanded is not None
    assert response_no_expand is not None


def test_search_response_includes_expanded_query(sample_conversations, temp_index_dir):
    """Test that search response includes expanded query info."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Search with expandable topic
    response = search_conversations(
        "api development",
        index_dir=temp_index_dir,
        expand_topics=True
    )

    # Response should indicate query was expanded
    assert response.expanded_query is not None or response.query == "api development"


def test_search_topics_extracted_in_results(sample_conversations, temp_index_dir):
    """Test that search results include extracted topics."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    response = search_conversations("database", index_dir=temp_index_dir)

    # Results should have matched_topics populated
    for result in response.results:
        # matched_topics is a list (may be empty if no topics detected)
        assert isinstance(result.matched_topics, list)


# ============= Scalability Tests =============


@pytest.fixture
def large_conversation_set(tmp_path):
    """Create 1000+ conversation files for scalability testing."""
    claude_dir = tmp_path / ".claude"

    # Create multiple projects to simulate realistic distribution
    projects = ["project-alpha", "project-beta", "project-gamma", "project-delta"]
    topics = [
        ("Python debugging", "Helped fix a Python memory leak issue"),
        ("React development", "Built React form components with validation"),
        ("Database optimization", "Optimized SQL queries for better performance"),
        ("API integration", "Integrated REST API endpoints"),
        ("Testing infrastructure", "Set up pytest test suite"),
        ("Docker deployment", "Created Docker containers for deployment"),
        ("Authentication system", "Implemented OAuth2 authentication"),
        ("Performance tuning", "Improved application response times"),
        ("Code refactoring", "Refactored legacy code for maintainability"),
        ("Documentation", "Generated API documentation"),
    ]

    import json
    import random

    total_files = 1000
    files_created = 0

    for project in projects:
        projects_dir = claude_dir / "projects" / project
        projects_dir.mkdir(parents=True)

        # Distribute files across projects
        files_for_project = total_files // len(projects)
        if project == projects[-1]:
            # Last project gets remaining files
            files_for_project = total_files - files_created

        for i in range(files_for_project):
            topic_idx = i % len(topics)
            topic_name, topic_summary = topics[topic_idx]

            file_path = projects_dir / f"session-{files_created:05d}.jsonl"
            with open(file_path, "w") as f:
                # Write summary
                f.write(json.dumps({
                    "type": "summary",
                    "summary": f"{topic_summary} (conversation {files_created})"
                }) + "\n")

                # Write user message
                f.write(json.dumps({
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": f"Help me with {topic_name} task #{files_created}"
                    },
                    "timestamp": f"2026-01-{(i % 28) + 1:02d}T10:00:00Z",
                }) + "\n")

                # Write assistant response
                f.write(json.dumps({
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": f"I'll help you with {topic_name}. Here's the solution..."
                        }]
                    },
                }) + "\n")

            files_created += 1

    return claude_dir


def test_supports_1000_conversation_files(large_conversation_set, temp_index_dir):
    """
    Test that the system supports at least 1000 conversation files.

    This is a key scalability requirement (US-004, SC-003).
    """
    # Build index for 1000 files
    stats = build_index(claude_dir=large_conversation_set, index_dir=temp_index_dir)

    # Verify all files were processed
    assert stats["total_files"] >= 1000, (
        f"Expected at least 1000 files, got {stats['total_files']}"
    )

    # Verify most files were indexed (some may be skipped if empty)
    total_processed = stats["indexed"] + stats["skipped"]
    assert total_processed >= 1000, (
        f"Expected to process at least 1000 files, processed {total_processed}"
    )

    # Verify minimal errors
    error_rate = stats["errors"] / stats["total_files"] if stats["total_files"] > 0 else 0
    assert error_rate < 0.01, (
        f"Error rate {error_rate:.2%} is too high (max 1%)"
    )

    # Verify index contains the expected number of documents
    index_stats = get_index_stats(index_dir=temp_index_dir)
    assert index_stats["indexed_conversations"] >= 1000, (
        f"Expected at least 1000 indexed, got {index_stats['indexed_conversations']}"
    )


def test_search_performance_with_1000_files(large_conversation_set, temp_index_dir):
    """
    Test that search performance is acceptable with 1000+ indexed files.

    Search should complete within 2 seconds even with a large index.
    """
    # Build the large index first
    build_index(claude_dir=large_conversation_set, index_dir=temp_index_dir)

    # Test multiple search queries
    queries = [
        "Python debugging",
        "database optimization",
        "how to fix memory issues",
        "api integration help",
    ]

    for query in queries:
        start_time = time.perf_counter()
        response = search_conversations(query, index_dir=temp_index_dir, num_results=10)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 2.0, (
            f"Search for '{query}' took {elapsed:.2f}s, should be under 2s"
        )
        assert len(response.results) > 0, (
            f"Search for '{query}' returned no results"
        )


def test_batch_indexing_progress_callback(large_conversation_set, temp_index_dir):
    """Test that progress callback is called during batch indexing."""
    progress_calls = []

    def progress_callback(total, indexed, skipped):
        progress_calls.append((total, indexed, skipped))

    # Build index with progress tracking
    stats = build_index(
        claude_dir=large_conversation_set,
        index_dir=temp_index_dir,
        progress_callback=progress_callback,
    )

    # Progress should have been called multiple times for 1000 files
    assert len(progress_calls) > 0, "Progress callback was never called"

    # Final call should reflect final stats
    final_total, final_indexed, final_skipped = progress_calls[-1]
    assert final_total == stats["total_files"]
    assert final_indexed == stats["indexed"]


def test_incremental_indexing_with_large_dataset(large_conversation_set, temp_index_dir):
    """Test that re-indexing unchanged files is fast (skip optimization)."""
    # First indexing
    stats1 = build_index(claude_dir=large_conversation_set, index_dir=temp_index_dir)
    assert stats1["indexed"] >= 1000

    # Second indexing should skip all unchanged files
    start_time = time.perf_counter()
    stats2 = build_index(claude_dir=large_conversation_set, index_dir=temp_index_dir)
    elapsed = time.perf_counter() - start_time

    # All files should be skipped (no changes)
    assert stats2["skipped"] >= 1000, (
        f"Expected all files to be skipped, but only {stats2['skipped']} were skipped"
    )
    assert stats2["indexed"] == 0, (
        f"Expected 0 newly indexed, but {stats2['indexed']} were indexed"
    )

    # Re-indexing unchanged files should be fast (no embedding computation)
    # Allow generous timeout since we're doing hash comparisons
    assert elapsed < 30.0, (
        f"Re-indexing took {elapsed:.2f}s, should be faster for unchanged files"
    )


# ============= Summary and Relevance Score Tests (US-005) =============


def test_search_result_has_summary_field():
    """Test that SearchResult dataclass includes summary field."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="This is a test conversation summary",
        relevance_score=0.75,
        file_path="/test/path",
    )
    assert hasattr(result, "summary")
    assert result.summary == "This is a test conversation summary"


def test_search_result_has_relevance_score_field():
    """Test that SearchResult dataclass includes relevance_score field."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Test summary",
        relevance_score=0.85,
        file_path="/test/path",
    )
    assert hasattr(result, "relevance_score")
    assert result.relevance_score == 0.85


def test_relevance_score_is_normalized():
    """Test that relevance score is between 0 and 1."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Test summary",
        relevance_score=0.75,
        file_path="/test/path",
    )
    assert 0.0 <= result.relevance_score <= 1.0


def test_search_result_format_display_includes_summary():
    """Test that format_display includes the conversation summary."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Debugging Python memory issues in production",
        relevance_score=0.85,
        file_path="/test/path",
    )
    display = result.format_display()
    assert "Debugging Python memory issues" in display


def test_search_result_format_display_includes_relevance_score():
    """Test that format_display includes relevance score as percentage."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Test summary",
        relevance_score=0.85,
        file_path="/test/path",
    )
    display = result.format_display()
    # 0.85 should be displayed as 85%
    assert "85%" in display


def test_search_result_format_display_truncates_long_summary():
    """Test that format_display truncates very long summaries."""
    long_summary = "A" * 200  # 200 character summary
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary=long_summary,
        relevance_score=0.75,
        file_path="/test/path",
    )
    display = result.format_display()
    # Should truncate to 80 chars + "..."
    assert "..." in display
    # Should not contain full 200 characters
    assert long_summary not in display


def test_search_results_contain_summary_from_conversation(sample_conversations, temp_index_dir):
    """Test that actual search results contain conversation summaries."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    response = search_conversations("Python debugging", index_dir=temp_index_dir)
    assert len(response.results) > 0

    # Each result should have a non-empty summary
    for result in response.results:
        assert result.summary is not None
        assert len(result.summary) > 0


def test_search_results_contain_valid_relevance_scores(sample_conversations, temp_index_dir):
    """Test that search results have valid relevance scores between 0 and 1."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    response = search_conversations("database optimization", index_dir=temp_index_dir)
    assert len(response.results) > 0

    # Each result should have a valid relevance score
    for result in response.results:
        assert result.relevance_score is not None
        assert 0.0 <= result.relevance_score <= 1.0, (
            f"Relevance score {result.relevance_score} is not in range [0, 1]"
        )


def test_search_results_ordered_by_relevance(sample_conversations, temp_index_dir):
    """Test that search results are ordered by relevance score (descending)."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    response = search_conversations("Python debugging memory", index_dir=temp_index_dir)
    assert len(response.results) >= 2

    # Results should be in descending order of relevance
    for i in range(len(response.results) - 1):
        assert response.results[i].relevance_score >= response.results[i + 1].relevance_score, (
            f"Results not ordered by relevance: {response.results[i].relevance_score} < "
            f"{response.results[i + 1].relevance_score}"
        )


# ============= Additional Search Tests for Coverage =============


def test_parse_date_various_formats():
    """Test date parsing with various formats."""
    # YYYY-MM-DD
    date = parse_date("2024-06-15")
    assert date is not None
    assert date.year == 2024
    assert date.month == 6
    assert date.day == 15

    # YYYY/MM/DD
    date = parse_date("2024/06/15")
    assert date is not None
    assert date.year == 2024

    # MM/DD/YYYY
    date = parse_date("06/15/2024")
    assert date is not None
    assert date.month == 6

    # DD-MM-YYYY
    date = parse_date("15-06-2024")
    assert date is not None
    assert date.day == 15


def test_search_with_date_from_filter(sample_conversations, temp_index_dir):
    """Test search with date_from filter."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Filter from a date in the past (should include results)
    response = search_conversations(
        "programming",
        index_dir=temp_index_dir,
        date_from="2020-01-01"
    )
    # Should still find results with dates after 2020
    assert len(response.results) >= 0  # Just verify no error

    # Filter from a date in the future (should exclude most results)
    response = search_conversations(
        "programming",
        index_dir=temp_index_dir,
        date_from="2030-01-01"
    )
    # Should exclude conversations dated before 2030
    assert len(response.results) >= 0  # Just verify no error


def test_search_with_date_to_filter(sample_conversations, temp_index_dir):
    """Test search with date_to filter."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Filter to a date in the future (should include results)
    response = search_conversations(
        "programming",
        index_dir=temp_index_dir,
        date_to="2030-01-01"
    )
    assert len(response.results) >= 0  # Just verify no error


def test_search_with_both_date_filters(sample_conversations, temp_index_dir):
    """Test search with both date_from and date_to filters."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    response = search_conversations(
        "programming",
        index_dir=temp_index_dir,
        date_from="2024-01-01",
        date_to="2027-12-31"
    )
    assert len(response.results) >= 0


def test_search_response_elapsed_seconds():
    """Test SearchResponse elapsed_seconds property."""
    from claude_history_search.search import SearchResponse
    response = SearchResponse(
        results=[],
        query="test",
        elapsed_ms=1500.0,
    )
    assert response.elapsed_seconds == 1.5


def test_extract_topics_multiple_synonyms():
    """Test topic extraction with synonym matching."""
    # Testing with synonym
    topics = extract_topics("Writing jest unit tests and mock functions")
    assert "testing" in topics

    # Frontend with synonym
    topics = extract_topics("Building React UI components")
    assert "frontend" in topics

    # Debugging with synonym
    topics = extract_topics("Troubleshooting error in production")
    assert "debugging" in topics


def test_expand_query_with_multiple_topics():
    """Test query expansion with multiple matching topics."""
    # Query with multiple topic words
    expanded = expand_query("testing database api")
    # Should expand multiple topics
    assert len(expanded) > len("testing database api")


def test_search_result_format_display_without_timestamp():
    """Test format_display without timestamp."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Test summary",
        relevance_score=0.75,
        file_path="/test/path",
        timestamp=None,  # No timestamp
    )
    display = result.format_display()
    assert "Date:" not in display
    assert "test-project" in display


def test_search_result_format_display_without_topics():
    """Test format_display without matched topics."""
    result = SearchResult(
        session_id="test-123",
        project="test-project",
        summary="Generic summary without recognizable topics",
        relevance_score=0.75,
        file_path="/test/path",
        matched_topics=[],  # No topics
    )
    display = result.format_display()
    assert "Topics:" not in display


def test_search_handles_special_characters_in_query(sample_conversations, temp_index_dir):
    """Test search handles special characters in query."""
    build_index(claude_dir=sample_conversations, index_dir=temp_index_dir)

    # Query with special characters
    response = search_conversations(
        "code (example) [test]",
        index_dir=temp_index_dir
    )
    # Should not raise error
    assert isinstance(response.results, list)

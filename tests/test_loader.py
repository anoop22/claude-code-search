"""Tests for conversation loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from claude_history_search.loader import (
    Conversation,
    find_conversation_files,
    load_conversations,
    parse_conversation_file,
)


@pytest.fixture
def temp_claude_dir():
    """Create a temporary Claude directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir = Path(tmpdir)
        projects_dir = claude_dir / "projects" / "test-project"
        projects_dir.mkdir(parents=True)
        yield claude_dir


def test_conversation_text_content():
    """Test Conversation.text_content property."""
    conv = Conversation(
        session_id="test-123",
        project="test-project",
        summaries=["Test summary"],
        messages=[
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ],
        timestamp=None,
        file_path=Path("/test"),
    )

    text = conv.text_content
    assert "Test summary" in text
    assert "Hello world" in text
    assert "Hi there!" in text


def test_conversation_summary_from_summaries():
    """Test that summary returns last summary."""
    conv = Conversation(
        session_id="test",
        project="test",
        summaries=["First summary", "Second summary", "Final summary"],
        messages=[],
        timestamp=None,
        file_path=Path("/test"),
    )

    assert conv.summary == "Final summary"


def test_conversation_summary_fallback():
    """Test summary falls back to first user message."""
    conv = Conversation(
        session_id="test",
        project="test",
        summaries=[],
        messages=[
            {"role": "user", "content": "This is my question about Python"},
        ],
        timestamp=None,
        file_path=Path("/test"),
    )

    assert "This is my question" in conv.summary


def test_find_conversation_files(temp_claude_dir):
    """Test finding conversation files."""
    projects_dir = temp_claude_dir / "projects" / "test-project"

    # Create test files
    (projects_dir / "session-001.jsonl").write_text("{}")
    (projects_dir / "session-002.jsonl").write_text("{}")
    (projects_dir / "agent-abc123.jsonl").write_text("{}")  # Should be skipped
    (projects_dir / "sessions-index.json").write_text("{}")  # Should be skipped

    files = list(find_conversation_files(temp_claude_dir))
    filenames = [f.name for f in files]

    assert "session-001.jsonl" in filenames
    assert "session-002.jsonl" in filenames
    assert "agent-abc123.jsonl" not in filenames
    assert "sessions-index.json" not in filenames


def test_parse_conversation_file(temp_claude_dir):
    """Test parsing a conversation file."""
    projects_dir = temp_claude_dir / "projects" / "test-project"
    file_path = projects_dir / "test-session.jsonl"

    # Create test conversation
    lines = [
        json.dumps({"type": "summary", "summary": "Test conversation"}),
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2026-01-15T10:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        }),
    ]
    file_path.write_text("\n".join(lines))

    conv = parse_conversation_file(file_path)

    assert conv is not None
    assert conv.session_id == "test-session"
    assert conv.project == "test-project"
    assert "Test conversation" in conv.summaries
    assert len(conv.messages) == 2


def test_parse_empty_file(temp_claude_dir):
    """Test parsing an empty file returns None."""
    projects_dir = temp_claude_dir / "projects" / "test-project"
    file_path = projects_dir / "empty.jsonl"
    file_path.write_text("")

    conv = parse_conversation_file(file_path)
    assert conv is None


def test_load_conversations(temp_claude_dir):
    """Test loading all conversations."""
    projects_dir = temp_claude_dir / "projects" / "test-project"

    # Create test conversations
    for i in range(3):
        file_path = projects_dir / f"session-{i:03d}.jsonl"
        lines = [
            json.dumps({"type": "summary", "summary": f"Conversation {i}"}),
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": f"Message {i}"},
            }),
        ]
        file_path.write_text("\n".join(lines))

    conversations = list(load_conversations(temp_claude_dir))
    assert len(conversations) == 3


# ============= Additional Edge Case Tests =============


def test_conversation_text_content_skips_meta_messages():
    """Test that meta messages starting with < or Caveat: are skipped."""
    conv = Conversation(
        session_id="test",
        project="test",
        summaries=[],
        messages=[
            {"role": "user", "content": "<system>system message</system>"},
            {"role": "user", "content": "Caveat: some caveat"},
            {"role": "user", "content": "Normal message"},
        ],
        timestamp=None,
        file_path=Path("/test"),
    )

    text = conv.text_content
    assert "<system>" not in text
    assert "Caveat:" not in text
    assert "Normal message" in text


def test_conversation_text_content_handles_empty_content():
    """Test handling of empty content in messages."""
    conv = Conversation(
        session_id="test",
        project="test",
        summaries=[],
        messages=[
            {"role": "user", "content": ""},
            {"role": "user"},  # Missing content key
            {"role": "assistant", "content": []},  # Empty list
        ],
        timestamp=None,
        file_path=Path("/test"),
    )

    text = conv.text_content
    assert text == ""


def test_conversation_text_content_handles_string_assistant_content():
    """Test handling assistant content as string instead of list."""
    conv = Conversation(
        session_id="test",
        project="test",
        summaries=[],
        messages=[
            {"role": "assistant", "content": "Plain string response"},
        ],
        timestamp=None,
        file_path=Path("/test"),
    )

    text = conv.text_content
    assert "Plain string response" in text


def test_conversation_summary_skips_meta_user_messages():
    """Test that summary fallback skips meta user messages."""
    conv = Conversation(
        session_id="test",
        project="test",
        summaries=[],
        messages=[
            {"role": "user", "content": "<meta>skip this</meta>"},
            {"role": "user", "content": "Use this message"},
        ],
        timestamp=None,
        file_path=Path("/test"),
    )

    assert conv.summary == "Use this message"


def test_parse_conversation_file_handles_missing_message_key():
    """Test parsing file where entry is missing message key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "test-project"
        projects_dir.mkdir()
        file_path = projects_dir / "test.jsonl"

        lines = [
            json.dumps({"type": "user"}),  # Missing message key
            json.dumps({"type": "summary", "summary": "Valid summary"}),
        ]
        file_path.write_text("\n".join(lines))

        conv = parse_conversation_file(file_path)
        assert conv is not None
        assert len(conv.summaries) == 1
        assert len(conv.messages) == 0


def test_parse_conversation_file_handles_empty_summary():
    """Test parsing file with empty summary string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "test-project"
        projects_dir.mkdir()
        file_path = projects_dir / "test.jsonl"

        lines = [
            json.dumps({"type": "summary", "summary": ""}),
            json.dumps({"type": "summary", "summary": "Valid summary"}),
        ]
        file_path.write_text("\n".join(lines))

        conv = parse_conversation_file(file_path)
        assert conv is not None
        # Empty summary should be filtered out
        assert conv.summaries == ["Valid summary"]


def test_find_conversation_files_handles_multiple_projects(temp_claude_dir):
    """Test finding files across multiple project directories."""
    # Create multiple projects
    for project in ["project-a", "project-b"]:
        project_dir = temp_claude_dir / "projects" / project
        project_dir.mkdir(parents=True)
        (project_dir / "session.jsonl").touch()

    files = list(find_conversation_files(temp_claude_dir))
    assert len(files) == 2

    # Both projects should be represented
    projects = {f.parent.name for f in files}
    assert "project-a" in projects
    assert "project-b" in projects


def test_conversation_with_timestamp_in_different_formats():
    """Test timestamp parsing with timezone designator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "test-project"
        projects_dir.mkdir()
        file_path = projects_dir / "test.jsonl"

        # ISO format with Z timezone
        lines = [
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "test"},
                "timestamp": "2024-01-15T10:30:00Z",
            }),
        ]
        file_path.write_text("\n".join(lines))

        conv = parse_conversation_file(file_path)
        assert conv is not None
        assert conv.timestamp is not None
        assert conv.timestamp.year == 2024
        assert conv.timestamp.month == 1
        assert conv.timestamp.day == 15


def test_load_conversations_with_default_dir():
    """Test that load_conversations uses default dir when None."""
    # This tests the branch where claude_dir is None
    # Won't find real conversations but tests the code path
    from claude_history_search.loader import get_default_claude_dir

    default_dir = get_default_claude_dir()
    # If default dir doesn't exist, should return empty iterator
    if not (default_dir / "projects").exists():
        convs = list(load_conversations(None))
        # Should complete without error
        assert isinstance(convs, list)

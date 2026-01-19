"""Load and parse Claude Code conversation history."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

DEBUG = False


@dataclass
class Conversation:
    """Represents a Claude Code conversation."""

    session_id: str
    project: str
    summaries: list[str]
    messages: list[dict]
    timestamp: datetime | None
    file_path: Path

    @property
    def text_content(self) -> str:
        """Get combined text content for embedding."""
        parts = []
        # Add summaries first (they're usually good descriptors)
        if self.summaries:
            parts.extend(self.summaries)
        # Add user and assistant message content
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    # Skip meta messages and command outputs
                    if not content.startswith("<") and not content.startswith("Caveat:"):
                        parts.append(content)
            elif msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                elif isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)

    @property
    def summary(self) -> str:
        """Get a summary of the conversation."""
        if self.summaries:
            return self.summaries[-1]  # Most recent summary
        # Fallback: use first user message
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content and not content.startswith("<"):
                    return content[:200] + ("..." if len(content) > 200 else "")
        return "No summary available"


def get_default_claude_dir() -> Path:
    """Get the default Claude Code directory."""
    return Path.home() / ".claude"


def find_conversation_files(claude_dir: Path | None = None) -> Iterator[Path]:
    """Find all conversation JSONL files in the Claude directory."""
    if claude_dir is None:
        claude_dir = get_default_claude_dir()

    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        if DEBUG:
            print(f"Projects directory not found: {projects_dir}")
        return

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for file_path in project_dir.glob("*.jsonl"):
            # Skip agent files and index files
            if file_path.name.startswith("agent-"):
                continue
            if file_path.name == "sessions-index.json":
                continue
            yield file_path


def parse_conversation_file(file_path: Path) -> Conversation | None:
    """Parse a single conversation JSONL file."""
    session_id = file_path.stem
    project = file_path.parent.name

    summaries = []
    messages = []
    timestamp = None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")

                # Extract summaries
                if entry_type == "summary":
                    summary = entry.get("summary", "")
                    if summary:
                        summaries.append(summary)

                # Extract messages
                elif entry_type in ("user", "assistant"):
                    msg = entry.get("message", {})
                    if msg:
                        messages.append(msg)
                    # Get timestamp from first message
                    if timestamp is None and "timestamp" in entry:
                        try:
                            ts_str = entry["timestamp"]
                            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass

        # Only return if we have meaningful content
        if not summaries and not messages:
            return None

        return Conversation(
            session_id=session_id,
            project=project,
            summaries=summaries,
            messages=messages,
            timestamp=timestamp,
            file_path=file_path,
        )
    except (OSError, PermissionError) as e:
        if DEBUG:
            print(f"Error reading {file_path}: {e}")
        return None


def load_conversations(claude_dir: Path | None = None) -> Iterator[Conversation]:
    """Load all conversations from the Claude directory."""
    for file_path in find_conversation_files(claude_dir):
        conversation = parse_conversation_file(file_path)
        if conversation:
            yield conversation

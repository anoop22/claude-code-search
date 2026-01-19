# Claude Code Search

Semantic search over your Claude Code conversation history. Find past implementations, recall discussions, and recover context from previous sessions.

## Why?

Claude Code stores conversations in `.jsonl` files at `~/.claude/projects/`. These files pile up quickly, and finding that conversation where you solved a tricky bug or implemented a feature becomes impossible. This tool indexes your conversation history and lets you search it semantically.

## Features

- **Semantic Search**: Find conversations by natural language ("how did I implement caching?")
- **Topic Detection**: Automatically recognizes programming topics (api, database, testing, auth, etc.)
- **Query Expansion**: Expands search terms with synonyms for better matching
- **Project Filtering**: Scope searches to specific projects
- **Date Filtering**: Search within date ranges
- **Latest Lookup**: Quickly find the most recent conversation in any project
- **Auto-Indexing**: Automatically indexes on first search - no manual setup needed
- **Incremental Updates**: Only re-indexes changed conversations
- **Scalable**: Handles 1000+ conversation files efficiently

## Installation

### From GitHub

```bash
# Clone the repo
git clone https://github.com/anoop22/claude-code-search.git
cd claude-code-search

# Install with pip
pip install .

# Or with pipx (recommended for CLI tools)
pipx install .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Just search - auto-indexes on first use
claude-search search "authentication implementation"

# Find latest conversation in a project
claude-search latest myproject

# Search by topic
claude-search topic database
```

## Usage

### Search Commands

```bash
# Semantic search
claude-search search "how to implement authentication"
claude-search search "fix memory leak"

# Topic-based search
claude-search topic api
claude-search topic database
claude-search topic testing

# Find latest conversation in a project
claude-search latest myapp
claude-search latest claude-code-search

# Filter by project
claude-search search "api endpoints" --project myapp

# Filter by date range
claude-search search "bug fix" --from 2024-01-01 --to 2024-06-30

# More results
claude-search search "query" -n 10

# Exact matching (disable query expansion)
claude-search search "db" --no-expand
```

### Index Management

```bash
# Check index status
claude-search status

# Manually update index
claude-search index

# Force complete rebuild
claude-search index --rebuild

# Custom batch size for large indexes
claude-search index --batch-size 50
```

## Search Results

Each result shows:
- **Relevance score** (0-100%): Color-coded - green (>70%), yellow (40-70%), red (<40%)
- **Summary**: Brief description of the conversation
- **Project**: Which project/directory the conversation belongs to
- **Session ID**: Unique identifier for finding the raw file
- **Date**: When the conversation occurred
- **Topics**: Detected programming topics

Example output:
```
Search results for: authentication
Found 3 results in 89ms

1. [78%] Implemented JWT authentication with refresh tokens
   Project: backend-api
   Session: a1b2c3d4-...
   Date: 2024-03-15
   Topics: authentication, api

2. [65%] Fixed login session timeout issue
   Project: webapp
   Session: e5f6g7h8-...
   Date: 2024-02-20
   Topics: authentication, debugging
```

## Recognized Topics

The search engine automatically detects these programming topics:

| Topic | Related Terms |
|-------|---------------|
| `api` | rest, endpoint, http, request, response, graphql |
| `database` | db, sql, postgres, mysql, sqlite, mongodb, query |
| `testing` | test, unittest, pytest, jest, spec, mock |
| `authentication` | auth, login, oauth, jwt, session, password |
| `frontend` | ui, react, vue, angular, css, html, component |
| `backend` | server, api, service, controller, handler |
| `debugging` | debug, bug, error, fix, issue, troubleshoot |
| `deployment` | deploy, ci/cd, docker, kubernetes, aws, cloud |
| `performance` | optimize, speed, fast, slow, cache, latency |
| `refactoring` | refactor, cleanup, restructure, improve |

## Claude Code Skill Integration

You can integrate this tool as a Claude Code skill for seamless access during conversations.

### Setup

1. Create the skill directory:
```bash
mkdir -p ~/.claude/skills/history-search
```

2. Create `~/.claude/skills/history-search/skill.md`:
```markdown
---
name: history-search
description: Search past Claude Code conversations semantically. Use when you need context from previous work.
---

# History Search

Search your Claude Code conversation history.

## Quick Reference

\`\`\`bash
# Semantic search
claude-search search "how to implement authentication"

# Topic search
claude-search topic database

# Find latest in project
claude-search latest myproject

# With filters
claude-search search "bug fix" --project myapp --from 2024-01-01
\`\`\`
```

3. Add to your `CLAUDE.md`:
```markdown
### Conversation History Search
Use the `/history-search` skill to find relevant past conversations:
- Before implementing features: Search for similar past implementations
- When debugging: Find how similar issues were resolved
- Starting on a project: Recover context from previous sessions
```

### Usage in Claude Code

Once configured, you can invoke the skill:
```
/history-search authentication
```

Or Claude will suggest using it when relevant context from past conversations would help.

## Architecture

```
~/.claude/
├── projects/                    # Claude Code conversations (source)
│   └── */
│       └── *.jsonl              # Conversation files
└── search_index/                # ChromaDB vector database (created by tool)
    └── chroma.sqlite3
```

**Components:**
- **Loader**: Parses JSONL conversation files, extracts text and metadata
- **Indexer**: Creates embeddings using sentence-transformers, stores in ChromaDB
- **Search**: Semantic vector search with filtering and ranking

## Requirements

- Python 3.10+
- ChromaDB (vector storage)
- sentence-transformers (embeddings)
- click (CLI)
- rich (terminal output)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=claude_history_search

# Format code
black src tests

# Lint
ruff check src tests
```

## How It Works

1. **Indexing**: Scans `~/.claude/projects/` for `.jsonl` conversation files
2. **Parsing**: Extracts human/assistant messages, timestamps, and project info
3. **Embedding**: Creates vector embeddings using `all-MiniLM-L6-v2` model
4. **Storage**: Stores embeddings and metadata in ChromaDB at `~/.claude/search_index`
5. **Search**: Performs cosine similarity search, filters results, ranks by relevance

The index uses content hashes to detect changes, so re-indexing only processes new or modified conversations.

## License

MIT

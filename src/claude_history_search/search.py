"""Search engine for Claude Code conversations."""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude_history_search.indexer import get_chroma_client, get_or_create_collection

DEBUG = False


@dataclass
class SearchResult:
    """A single search result."""

    session_id: str
    project: str
    summary: str
    relevance_score: float
    file_path: str
    timestamp: str | None = None
    matched_topics: list[str] = field(default_factory=list)

    def format_display(self) -> str:
        """Format result for CLI display."""
        score_pct = int(self.relevance_score * 100)
        lines = [
            f"[{score_pct}%] {self.summary[:80]}{'...' if len(self.summary) > 80 else ''}",
            f"    Project: {self.project}",
            f"    Session: {self.session_id}",
        ]
        if self.timestamp:
            lines.append(f"    Date: {self.timestamp[:10]}")
        if self.matched_topics:
            lines.append(f"    Topics: {', '.join(self.matched_topics)}")
        return "\n".join(lines)


# Common programming topics and their related terms for query expansion
TOPIC_SYNONYMS = {
    "api": ["rest", "endpoint", "http", "request", "response", "graphql"],
    "database": ["db", "sql", "postgres", "mysql", "sqlite", "mongodb", "query"],
    "testing": ["test", "unittest", "pytest", "jest", "spec", "mock", "assertion"],
    "debugging": ["debug", "bug", "error", "fix", "issue", "problem", "troubleshoot"],
    "frontend": ["ui", "react", "vue", "angular", "css", "html", "component"],
    "backend": ["server", "api", "service", "controller", "handler"],
    "deployment": ["deploy", "ci/cd", "docker", "kubernetes", "k8s", "aws", "cloud"],
    "authentication": ["auth", "login", "oauth", "jwt", "session", "password"],
    "performance": ["optimize", "speed", "fast", "slow", "cache", "latency"],
    "refactoring": ["refactor", "cleanup", "restructure", "improve", "rewrite"],
}


@dataclass
class SearchResponse:
    """Response from a search query."""

    results: list[SearchResult]
    query: str
    elapsed_ms: float
    expanded_query: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000


def extract_topics(text: str) -> list[str]:
    """Extract recognized topics from text."""
    text_lower = text.lower()
    found_topics = []

    for topic, synonyms in TOPIC_SYNONYMS.items():
        if topic in text_lower or any(syn in text_lower for syn in synonyms):
            found_topics.append(topic)

    return found_topics


def expand_query(query: str) -> str:
    """
    Expand a query with related terms for better semantic matching.

    This helps when users search for topics using different terminology
    than what's in the conversation content.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    expansions = []

    # Find matching topics and add their synonyms
    for topic, synonyms in TOPIC_SYNONYMS.items():
        # Check if topic word is in query (word boundary match)
        if topic in query_words:
            # Add a few relevant synonyms
            expansions.extend(synonyms[:3])
        else:
            # Check if any synonym is in the query (word boundary match)
            for syn in synonyms:
                if syn in query_words:
                    expansions.append(topic)
                    break

    if expansions:
        # Combine original query with expansions
        return f"{query} {' '.join(set(expansions))}"

    return query


def parse_date(date_str: str) -> datetime | None:
    """Parse a date string in various formats."""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def search_conversations(
    query: str,
    num_results: int = 5,
    index_dir: Path | None = None,
    project_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    expand_topics: bool = True,
) -> SearchResponse:
    """
    Search conversations by semantic query (topic or natural language).

    Args:
        query: The search query (natural language or topic)
        num_results: Maximum number of results to return
        index_dir: Optional custom index directory
        project_filter: Filter results by project name (partial match)
        date_from: Filter results from this date (YYYY-MM-DD)
        date_to: Filter results until this date (YYYY-MM-DD)
        expand_topics: Whether to expand query with topic synonyms

    Returns:
        SearchResponse with results and timing info
    """
    start_time = time.perf_counter()

    client = get_chroma_client(index_dir)
    collection = get_or_create_collection(client)

    # Check if index is empty
    if collection.count() == 0:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SearchResponse(results=[], query=query, elapsed_ms=elapsed_ms)

    # Expand query with topic synonyms for better matching
    search_query = expand_query(query) if expand_topics else query
    expanded = search_query if search_query != query else None

    if DEBUG and expanded:
        print(f"Expanded query: {query} -> {search_query}")

    # Build ChromaDB where clause for filtering
    where_clause = None
    if project_filter:
        # ChromaDB doesn't support LIKE, so we'll filter after
        pass

    # Request more results than needed to allow for filtering
    fetch_count = min(num_results * 3, collection.count()) if project_filter or date_from or date_to else min(num_results, collection.count())

    # Perform semantic search
    results = collection.query(
        query_texts=[search_query],
        n_results=fetch_count,
        include=["metadatas", "distances"],
    )

    # Parse date filters
    date_from_dt = parse_date(date_from) if date_from else None
    date_to_dt = parse_date(date_to) if date_to else None

    search_results = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0

            # Apply project filter
            if project_filter:
                project = metadata.get("project", "")
                if project_filter.lower() not in project.lower():
                    continue

            # Apply date filters
            timestamp_str = metadata.get("timestamp")
            if timestamp_str and (date_from_dt or date_to_dt):
                try:
                    ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    ts_date = ts.replace(tzinfo=None)
                    if date_from_dt and ts_date < date_from_dt:
                        continue
                    if date_to_dt and ts_date > date_to_dt:
                        continue
                except (ValueError, TypeError):
                    pass  # Keep results with unparseable timestamps

            # Convert cosine distance to similarity score (0-1)
            # ChromaDB returns distance, lower is better
            # For cosine distance: similarity = 1 - distance
            relevance_score = max(0.0, min(1.0, 1.0 - distance))

            # Extract matched topics from the summary
            summary = metadata.get("summary", "No summary")
            matched_topics = extract_topics(summary)

            search_results.append(
                SearchResult(
                    session_id=doc_id,
                    project=metadata.get("project", "unknown"),
                    summary=summary,
                    relevance_score=relevance_score,
                    file_path=metadata.get("file_path", ""),
                    timestamp=timestamp_str,
                    matched_topics=matched_topics,
                )
            )

            # Stop once we have enough filtered results
            if len(search_results) >= num_results:
                break

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if DEBUG:
        print(f"Search completed in {elapsed_ms:.2f}ms")

    return SearchResponse(
        results=search_results,
        query=query,
        elapsed_ms=elapsed_ms,
        expanded_query=expanded,
    )


def get_latest_by_project(
    project_filter: str,
    index_dir: Path | None = None,
) -> SearchResult | None:
    """
    Get the latest conversation for a specific project.

    Args:
        project_filter: Project name to filter by (partial match)
        index_dir: Optional custom index directory

    Returns:
        The most recent SearchResult or None if no matches
    """
    start_time = time.perf_counter()

    client = get_chroma_client(index_dir)
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return None

    # Get all documents to filter by project
    # ChromaDB doesn't support LIKE queries, so we fetch and filter
    all_docs = collection.get(include=["metadatas"])

    if not all_docs["ids"]:
        return None

    # Filter by project and find the latest by timestamp
    matching_results = []
    for i, doc_id in enumerate(all_docs["ids"]):
        metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
        project = metadata.get("project", "")

        if project_filter.lower() in project.lower():
            timestamp_str = metadata.get("timestamp")
            matching_results.append((doc_id, metadata, timestamp_str))

    if not matching_results:
        return None

    # Sort by timestamp descending (latest first)
    def sort_key(item):
        ts = item[2]
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                # Convert to naive datetime for comparison
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except (ValueError, TypeError):
                pass
        return datetime.min

    matching_results.sort(key=sort_key, reverse=True)

    # Return the latest one
    doc_id, metadata, timestamp_str = matching_results[0]
    summary = metadata.get("summary", "No summary")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if DEBUG:
        print(f"Latest lookup completed in {elapsed_ms:.2f}ms")

    return SearchResult(
        session_id=doc_id,
        project=metadata.get("project", "unknown"),
        summary=summary,
        relevance_score=1.0,
        file_path=metadata.get("file_path", ""),
        timestamp=timestamp_str,
        matched_topics=extract_topics(summary),
    )


def search_by_topic(
    topic: str,
    num_results: int = 5,
    index_dir: Path | None = None,
) -> SearchResponse:
    """
    Search conversations by a specific topic.

    This is a convenience wrapper that ensures topic expansion is enabled
    and may apply additional topic-specific processing.

    Args:
        topic: The topic to search for
        num_results: Maximum number of results to return
        index_dir: Optional custom index directory

    Returns:
        SearchResponse with results and timing info
    """
    # Create a topic-focused query
    topic_query = f"conversations about {topic}"

    return search_conversations(
        query=topic_query,
        num_results=num_results,
        index_dir=index_dir,
        expand_topics=True,
    )

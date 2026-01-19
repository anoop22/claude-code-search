"""CLI entry point for Claude History Search."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from claude_history_search import __version__
from claude_history_search.indexer import build_index, get_index_stats
from claude_history_search.search import search_conversations

console = Console()


def ensure_indexed() -> bool:
    """
    Ensure conversations are indexed before searching.

    If no conversations are indexed, automatically runs indexing.
    Returns True if ready to search, False if no conversations found.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    stats = get_index_stats()
    if stats.get("indexed_conversations", 0) > 0:
        return True

    # Auto-index
    console.print("[yellow]No conversations indexed. Indexing now...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Indexing conversations...", total=None)
        build_stats = build_index()

    if build_stats["indexed"] > 0:
        console.print(f"[green]Indexed {build_stats['indexed']} conversations.[/green]\n")
        return True
    elif build_stats["total_files"] == 0:
        console.print("[red]No conversation files found.[/red]")
        return False
    else:
        console.print("[yellow]No new conversations to index.[/yellow]")
        return get_index_stats().get("indexed_conversations", 0) > 0


@click.group()
@click.version_option(version=__version__, prog_name="claude-search")
def main():
    """Semantic search over Claude Code conversation history."""
    pass


@main.command()
@click.argument("query")
@click.option("-n", "--num-results", default=5, help="Number of results to return")
@click.option("-p", "--project", default=None, help="Filter by project name (partial match)")
@click.option("--from", "date_from", default=None, help="Filter from date (YYYY-MM-DD)")
@click.option("--to", "date_to", default=None, help="Filter to date (YYYY-MM-DD)")
@click.option("--no-expand", is_flag=True, help="Disable topic query expansion")
def search(query: str, num_results: int, project: str, date_from: str, date_to: str, no_expand: bool):
    """Search conversations by topic or natural language query.

    Examples:

    \b
        claude-search search "database migrations"
        claude-search search "how to fix authentication"
        claude-search search "api endpoint" --project myapp
        claude-search search "testing" --from 2024-01-01
    """
    # Ensure index exists (auto-index if needed)
    if not ensure_indexed():
        return

    # Perform search
    response = search_conversations(
        query,
        num_results=num_results,
        project_filter=project,
        date_from=date_from,
        date_to=date_to,
        expand_topics=not no_expand,
    )

    if not response.results:
        console.print(f"[yellow]No results found for:[/yellow] {query}")
        if project:
            console.print(f"[dim]Project filter: {project}[/dim]")
        return

    # Display results header
    console.print(f"\n[bold]Search results for:[/bold] {query}")
    if response.expanded_query:
        console.print(f"[dim]Query expanded with related terms[/dim]")

    filter_info = []
    if project:
        filter_info.append(f"project={project}")
    if date_from:
        filter_info.append(f"from={date_from}")
    if date_to:
        filter_info.append(f"to={date_to}")

    timing_info = f"Found {len(response.results)} results in {response.elapsed_ms:.0f}ms"
    if filter_info:
        timing_info += f" (filters: {', '.join(filter_info)})"
    console.print(f"[dim]{timing_info}[/dim]\n")

    for i, result in enumerate(response.results, 1):
        score_color = "green" if result.relevance_score > 0.7 else (
            "yellow" if result.relevance_score > 0.4 else "red"
        )
        score_pct = int(result.relevance_score * 100)

        console.print(f"[bold]{i}.[/bold] [{score_color}][{score_pct}%][/{score_color}] {result.summary}")
        console.print(f"   [dim]Project:[/dim] {result.project}")
        console.print(f"   [dim]Session:[/dim] {result.session_id}")
        if result.timestamp:
            console.print(f"   [dim]Date:[/dim] {result.timestamp[:10]}")
        if result.matched_topics:
            console.print(f"   [dim]Topics:[/dim] {', '.join(result.matched_topics)}")
        console.print()


@main.command()
@click.argument("topic")
@click.option("-n", "--num-results", default=5, help="Number of results to return")
def topic(topic: str, num_results: int):
    """Search conversations by topic.

    This is optimized for finding conversations about a specific subject area.

    Examples:

    \b
        claude-search topic api
        claude-search topic "database optimization"
        claude-search topic testing -n 10
    """
    from claude_history_search.search import search_by_topic

    # Ensure index exists (auto-index if needed)
    if not ensure_indexed():
        return

    # Perform topic search
    response = search_by_topic(topic, num_results=num_results)

    if not response.results:
        console.print(f"[yellow]No conversations found about:[/yellow] {topic}")
        return

    # Display results
    console.print(f"\n[bold]Conversations about:[/bold] {topic}")
    console.print(f"[dim]Found {len(response.results)} results in {response.elapsed_ms:.0f}ms[/dim]\n")

    for i, result in enumerate(response.results, 1):
        score_color = "green" if result.relevance_score > 0.7 else (
            "yellow" if result.relevance_score > 0.4 else "red"
        )
        score_pct = int(result.relevance_score * 100)

        console.print(f"[bold]{i}.[/bold] [{score_color}][{score_pct}%][/{score_color}] {result.summary}")
        console.print(f"   [dim]Project:[/dim] {result.project}")
        console.print(f"   [dim]Session:[/dim] {result.session_id}")
        if result.timestamp:
            console.print(f"   [dim]Date:[/dim] {result.timestamp[:10]}")
        if result.matched_topics:
            console.print(f"   [dim]Topics:[/dim] {', '.join(result.matched_topics)}")
        console.print()


@main.command()
@click.argument("project")
def latest(project: str):
    """Show the latest conversation in a project/folder.

    Examples:

    \b
        claude-search latest myapp
        claude-search latest claude-history-search
    """
    from claude_history_search.search import get_latest_by_project

    # Ensure index exists (auto-index if needed)
    if not ensure_indexed():
        return

    # Get latest conversation for project
    result = get_latest_by_project(project)

    if not result:
        console.print(f"[yellow]No conversations found for project:[/yellow] {project}")
        return

    # Display result
    console.print(f"\n[bold]Latest conversation in:[/bold] {project}\n")
    console.print(f"[bold]Summary:[/bold] {result.summary}")
    console.print(f"[dim]Project:[/dim] {result.project}")
    console.print(f"[dim]Session:[/dim] {result.session_id}")
    if result.timestamp:
        console.print(f"[dim]Date:[/dim] {result.timestamp[:10]}")
    if result.matched_topics:
        console.print(f"[dim]Topics:[/dim] {', '.join(result.matched_topics)}")
    console.print()


@main.command()
@click.option("--path", default=None, help="Path to Claude Code conversations directory")
@click.option("--rebuild", is_flag=True, help="Force rebuild the entire index")
@click.option("--batch-size", default=100, help="Batch size for indexing (default: 100)")
def index(path: str, rebuild: bool, batch_size: int):
    """Build or update the search index.

    Supports indexing 1000+ conversation files with progress tracking.
    Uses batch processing for efficient indexing of large conversation histories.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    claude_dir = Path(path) if path else None

    if rebuild:
        from claude_history_search.indexer import clear_index
        console.print("[yellow]Clearing existing index...[/yellow]")
        clear_index()

    console.print("[bold]Indexing conversations...[/bold]")

    # Use progress bar for large datasets
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            "Indexing...",
            total=None,  # Unknown total initially
            status="Scanning..."
        )

        last_total = [0]  # Mutable container for closure

        def progress_callback(total: int, indexed: int, skipped: int):
            # Update progress bar
            if last_total[0] == 0 and total > 0:
                # First callback - we now know the approximate scale
                progress.update(task_id, total=total)
            progress.update(
                task_id,
                completed=indexed + skipped,
                status=f"Indexed: {indexed}, Skipped: {skipped}"
            )
            last_total[0] = total

        stats = build_index(
            claude_dir=claude_dir,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

    # Display results
    table = Table(title="Indexing Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Total files scanned", str(stats["total_files"]))
    table.add_row("Newly indexed", str(stats["indexed"]))
    table.add_row("Unchanged (skipped)", str(stats["skipped"]))
    if stats["errors"] > 0:
        table.add_row("Errors", f"[red]{stats['errors']}[/red]")

    console.print(table)

    # Summary message for large datasets
    if stats["total_files"] >= 100:
        console.print(
            f"\n[dim]Processed {stats['total_files']} files in batches of {batch_size}[/dim]"
        )


@main.command()
def status():
    """Show index status and statistics."""
    stats = get_index_stats()

    if stats.get("error"):
        console.print(f"[red]Error:[/red] {stats['error']}")
        return

    table = Table(title="Index Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Indexed conversations", str(stats["indexed_conversations"]))
    table.add_row("Index location", stats.get("index_path", "unknown"))

    console.print(table)

    if stats["indexed_conversations"] == 0:
        console.print("\n[yellow]No conversations indexed yet.[/yellow]")
        console.print("Run 'claude-search index' to build the search index.")


if __name__ == "__main__":
    main()

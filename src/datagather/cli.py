"""Command-line interface for DataGather."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from datagather import __version__
from datagather.config import GatherConfig
from datagather.storage.checkpoint import CheckpointManager
from datagather.storage.jsonl import JSONLWriter
from datagather.utils.chunking import chunk_document

# Create Typer app
app = typer.Typer(
    name="datagather",
    help="Gather diverse English text data from open sources for NLP research.",
    add_completion=False,
)

# Create subcommands
sources_app = typer.Typer(help="Manage data sources")
app.add_typer(sources_app, name="sources")

# Console for rich output
console = Console()

# Source registry - maps source names to their classes
SOURCE_REGISTRY = {}


def register_sources():
    """Register all available sources."""
    global SOURCE_REGISTRY

    # Import sources here to avoid circular imports
    from datagather.sources.wikipedia import WikipediaSource

    SOURCE_REGISTRY = {
        "wikipedia": WikipediaSource,
    }

    # Try to import optional sources
    try:
        from datagather.sources.gutenberg import GutenbergSource

        SOURCE_REGISTRY["gutenberg"] = GutenbergSource
    except ImportError:
        pass

    try:
        from datagather.sources.arxiv import ArxivSource

        SOURCE_REGISTRY["arxiv"] = ArxivSource
    except ImportError:
        pass

    try:
        from datagather.sources.code import CodeSource

        SOURCE_REGISTRY["code"] = CodeSource
    except ImportError:
        pass

    try:
        from datagather.sources.reddit import RedditSource

        SOURCE_REGISTRY["reddit"] = RedditSource
    except ImportError:
        pass

    try:
        from datagather.sources.news import NewsSource

        SOURCE_REGISTRY["news"] = NewsSource
    except ImportError:
        pass

    try:
        from datagather.sources.legal import LegalSource

        SOURCE_REGISTRY["legal"] = LegalSource
    except ImportError:
        pass

    try:
        from datagather.sources.conversational import ConversationalSource

        SOURCE_REGISTRY["conversational"] = ConversationalSource
    except ImportError:
        pass

    try:
        from datagather.sources.twitter import TwitterSource

        SOURCE_REGISTRY["twitter"] = TwitterSource
    except ImportError:
        pass


def setup_logging(verbose: bool = False):
    """Set up logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_config(config_path: Optional[Path]) -> GatherConfig:
    """Load configuration from file or use defaults."""
    if config_path and config_path.exists():
        console.print(f"Loading config from [cyan]{config_path}[/cyan]")
        return GatherConfig.from_yaml(config_path)
    return GatherConfig.default()


@app.command()
def gather(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML)",
    ),
    samples_per_source: Optional[int] = typer.Option(
        None,
        "--samples-per-source",
        "-n",
        help="Number of samples per source (overrides config)",
    ),
    sources: Optional[str] = typer.Option(
        None,
        "--sources",
        "-s",
        help="Comma-separated list of sources to run",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (overrides config)",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from checkpoint if available",
    ),
    skip_full_docs: bool = typer.Option(
        False,
        "--skip-full-docs",
        help="Don't save full documents, only chunks (saves disk space)",
    ),
    chunk_size: Optional[int] = typer.Option(
        None,
        "--chunk-size",
        help="Token count per chunk (overrides config)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Gather text data from configured sources.

    Examples:
        datagather gather --samples-per-source 100
        datagather gather --sources wikipedia,arxiv --samples-per-source 50
        datagather gather --config custom_config.yaml --skip-full-docs
    """
    setup_logging(verbose)
    register_sources()

    # Load configuration
    cfg = load_config(config)

    # Apply overrides
    if output_dir:
        cfg.output_dir = str(output_dir)
    if chunk_size:
        cfg.chunk_size = chunk_size
    if skip_full_docs:
        cfg.save_full_documents = False

    # Determine which sources to run
    if sources:
        source_list = [s.strip() for s in sources.split(",")]
    else:
        source_list = cfg.get_enabled_sources()

    # Filter to available sources
    available_sources = [s for s in source_list if s in SOURCE_REGISTRY]
    missing_sources = [s for s in source_list if s not in SOURCE_REGISTRY]

    if missing_sources:
        console.print(
            f"[yellow]Warning: Sources not available: {', '.join(missing_sources)}[/yellow]"
        )

    if not available_sources:
        console.print("[red]Error: No sources available to run[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]DataGather v{__version__}[/bold]")
    console.print(f"Output directory: [cyan]{cfg.output_dir}[/cyan]")
    console.print(f"Sources to run: [cyan]{', '.join(available_sources)}[/cyan]\n")

    # Create output directories
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize storage
    writer = JSONLWriter(output_path)
    checkpoint_mgr = CheckpointManager(Path(cfg.checkpoint.directory))

    total_docs = 0
    total_chunks = 0

    try:
        for source_name in available_sources:
            # Get source configuration
            source_config = getattr(cfg.sources, source_name)
            if not source_config.enabled:
                console.print(f"[dim]Skipping {source_name} (disabled)[/dim]")
                continue

            # Determine sample count
            target_samples = samples_per_source or source_config.samples

            console.print(f"\n[bold blue]Processing {source_name}...[/bold blue]")

            # Load checkpoint if resuming
            checkpoint_state = None
            if resume:
                checkpoint_state = checkpoint_mgr.load(source_name)
                if checkpoint_state:
                    completed = checkpoint_state.get("completed_samples", 0)
                    console.print(f"  Resuming from checkpoint: {completed} samples completed")
                    target_samples = max(0, target_samples - completed)
                    if target_samples == 0:
                        console.print(f"  [green]Already completed![/green]")
                        continue

            # Initialize source
            source_class = SOURCE_REGISTRY[source_name]
            source = source_class(source_config, checkpoint_state)

            # Validate config
            errors = source.validate_config()
            if errors:
                console.print(f"  [red]Config errors: {', '.join(errors)}[/red]")
                continue

            source.initialize()
            checkpoint_mgr.mark_started(source_name, target_samples)

            # Gather documents with progress
            source_docs = 0
            source_chunks = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Gathering {source_name}", total=target_samples)

                try:
                    for doc in source.fetch_documents(target_samples):
                        # Write full document if not skipping
                        if cfg.save_full_documents:
                            writer.write_document(doc)

                        # Create and write chunks
                        if cfg.save_chunks:
                            chunks = chunk_document(
                                doc,
                                strategy=cfg.chunking_strategy,
                                chunk_size=cfg.chunk_size,
                                overlap=cfg.chunk_overlap,
                            )
                            for chunk in chunks:
                                writer.write_chunk(chunk)
                            source_chunks += len(chunks)

                        source_docs += 1
                        progress.update(task, advance=1)

                        # Save checkpoint periodically
                        if checkpoint_mgr.should_save(source_docs, cfg.checkpoint.save_interval):
                            checkpoint_mgr.save(
                                source_name,
                                {
                                    **source.get_checkpoint_state(),
                                    "completed_samples": source_docs,
                                    "target_samples": target_samples,
                                },
                            )

                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted! Saving checkpoint...[/yellow]")
                    checkpoint_mgr.save(
                        source_name,
                        {
                            **source.get_checkpoint_state(),
                            "completed_samples": source_docs,
                            "target_samples": target_samples,
                            "status": "interrupted",
                        },
                    )
                    raise

                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    checkpoint_mgr.mark_failed(source_name, str(e))
                    continue

            # Mark completed and cleanup
            source.cleanup()
            checkpoint_mgr.mark_completed(source_name)

            console.print(
                f"  [green]Completed: {source_docs} documents, {source_chunks} chunks[/green]"
            )
            total_docs += source_docs
            total_chunks += source_chunks

    finally:
        writer.close()

    # Summary
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"Total documents: {total_docs}")
    console.print(f"Total chunks: {total_chunks}")
    console.print(f"Output: {cfg.output_dir}")


@sources_app.command("list")
def list_sources():
    """List available data sources."""
    register_sources()

    table = Table(title="Available Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Available", style="green")
    table.add_column("Description")

    source_info = {
        "wikipedia": "Random Wikipedia articles",
        "gutenberg": "Project Gutenberg books (public domain)",
        "code": "Code from The Stack/CodeParrot (HuggingFace)",
        "arxiv": "Academic papers from arXiv",
        "reddit": "Reddit posts/comments (Pushshift archives)",
        "twitter": "Twitter data (GESIS TweetsKB archives)",
        "news": "News articles (Common Crawl)",
        "legal": "Legal documents (Pile of Law, USPTO)",
        "conversational": "Conversational data (Oyez, dialog datasets)",
    }

    for name, desc in source_info.items():
        available = "Yes" if name in SOURCE_REGISTRY else "No"
        table.add_row(name, available, desc)

    console.print(table)


@sources_app.command("info")
def source_info(
    source_name: str = typer.Argument(..., help="Source name"),
):
    """Show detailed information about a source."""
    register_sources()

    if source_name not in SOURCE_REGISTRY:
        console.print(f"[red]Unknown source: {source_name}[/red]")
        raise typer.Exit(1)

    source_class = SOURCE_REGISTRY[source_name]
    console.print(f"\n[bold]{source_name}[/bold]")
    console.print(f"Class: {source_class.__module__}.{source_class.__name__}")
    if source_class.__doc__:
        console.print(f"\n{source_class.__doc__}")


@app.command()
def status(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Show current checkpoint status for all sources."""
    cfg = load_config(config)
    checkpoint_mgr = CheckpointManager(Path(cfg.checkpoint.directory))

    checkpoints = checkpoint_mgr.list_checkpoints()

    if not checkpoints:
        console.print("[dim]No checkpoints found[/dim]")
        return

    table = Table(title="Checkpoint Status")
    table.add_column("Source", style="cyan")
    table.add_column("Status")
    table.add_column("Completed")
    table.add_column("Target")
    table.add_column("Errors")
    table.add_column("Last Updated")

    for cp in checkpoints:
        status_style = {
            "completed": "green",
            "in_progress": "yellow",
            "failed": "red",
            "interrupted": "orange1",
        }.get(cp["status"], "white")

        table.add_row(
            cp["source"],
            f"[{status_style}]{cp['status']}[/{status_style}]",
            str(cp["completed_samples"]),
            str(cp.get("target_samples", "-")),
            str(cp.get("error_count", 0)),
            cp.get("updated_at", "-")[:19] if cp.get("updated_at") else "-",
        )

    console.print(table)


@app.command()
def reset(
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Source to reset (all if not specified)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
):
    """Reset checkpoints (allows re-running sources)."""
    cfg = load_config(config)
    checkpoint_mgr = CheckpointManager(Path(cfg.checkpoint.directory))

    if source:
        msg = f"Reset checkpoint for {source}?"
    else:
        msg = "Reset ALL checkpoints?"

    if not force:
        if not typer.confirm(msg):
            raise typer.Abort()

    count = checkpoint_mgr.reset(source)
    console.print(f"[green]Reset {count} checkpoint(s)[/green]")


@app.command()
def convert(
    input_dir: Path = typer.Option(
        Path("./data/raw"),
        "--input",
        "-i",
        help="Input directory with JSONL files",
    ),
    output_file: Path = typer.Option(
        Path("./data/parquet/all_data.parquet"),
        "--output",
        "-o",
        help="Output Parquet file",
    ),
):
    """Convert JSONL files to Parquet format."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError:
        console.print("[red]pandas and pyarrow required for conversion[/red]")
        raise typer.Exit(1)

    from datagather.storage.jsonl import iter_jsonl

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    # Collect all records
    records = []
    for jsonl_file in input_dir.glob("*.jsonl"):
        console.print(f"Reading {jsonl_file.name}...")
        for record in iter_jsonl(jsonl_file):
            records.append(record)

    if not records:
        console.print("[yellow]No records found[/yellow]")
        return

    console.print(f"Converting {len(records)} records...")

    # Create DataFrame and save as Parquet
    df = pd.DataFrame(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file)

    console.print(f"[green]Saved to {output_file}[/green]")


@app.command()
def version():
    """Show version information."""
    console.print(f"DataGather v{__version__}")


@app.callback()
def main():
    """DataGather - Gather diverse English text data from open sources."""
    pass


if __name__ == "__main__":
    app()

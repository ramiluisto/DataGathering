# DataGather

Gather diverse English text data from open sources for NLP research, particularly for studying LLM and embedding model anisotropy.

## Features

- **9 Data Sources**: Wikipedia, Project Gutenberg, Code (GitHub), arXiv, Reddit, Twitter, News, Legal documents, and Conversational transcripts
- **Configurable**: YAML config file for fine-grained control, CLI flags for quick overrides
- **Resumable**: Checkpoint system for long-running scrapes
- **Flexible Output**: Full documents and/or pre-chunked snippets (token/sentence/paragraph-based)
- **Research-Friendly**: All sources are open/permissively licensed for academic use

## Installation

```bash
# Clone the repository
git clone https://github.com/ramiluisto/DataGathering.git
cd DataGathering

# Install with UV (recommended)
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Gather 100 samples from each enabled source
datagather gather --samples-per-source 100

# Gather from specific sources only
datagather gather --sources wikipedia,arxiv,gutenberg --samples-per-source 50

# Use a custom config file
datagather gather --config my_config.yaml

# Save disk space (skip full documents, keep only chunks)
datagather gather --skip-full-docs
```

## CLI Commands

```bash
datagather gather [OPTIONS]    # Main data gathering command
datagather sources list        # List available sources
datagather sources info NAME   # Show details about a source
datagather status              # Show checkpoint status
datagather reset               # Reset checkpoints
datagather convert             # Convert JSONL to Parquet
datagather version             # Show version
```

### Gather Options

| Flag | Description |
|------|-------------|
| `--config, -c PATH` | Path to YAML config file |
| `--samples-per-source, -n INT` | Number of samples per source |
| `--sources, -s TEXT` | Comma-separated list of sources |
| `--output-dir, -o PATH` | Output directory |
| `--resume/--no-resume` | Resume from checkpoint (default: resume) |
| `--skip-full-docs` | Only save chunks (saves disk space) |
| `--chunk-size INT` | Tokens per chunk (default: 512) |
| `--verbose, -v` | Enable verbose logging |

## Data Sources

| Source | Description | License | Notes |
|--------|-------------|---------|-------|
| **wikipedia** | Random Wikipedia articles | Fair use | MediaWiki API |
| **gutenberg** | Public domain books | Public domain | Via Gutendex API |
| **code** | GitHub code (Python, JS, etc.) | Permissive | The Stack / CodeParrot |
| **arxiv** | Academic paper abstracts | CC-BY | Via arxiv package |
| **reddit** | Reddit posts/comments | Various | HuggingFace datasets |
| **twitter** | Archived tweets | Various | HuggingFace datasets |
| **news** | News articles | Fair use | CC-News, CNN/DailyMail |
| **legal** | Legal documents | Public domain | Pile of Law |
| **conversational** | Dialog datasets | Open | DailyDialog, TopicalChat |

## Configuration

Create a `config.yaml` file (see `config.yaml` in repo for full example):

```yaml
output_dir: "./data"
default_samples_per_source: 100
save_full_documents: true
save_chunks: true
chunk_size: 512
chunking_strategy: "token"  # token, sentence, paragraph

sources:
  wikipedia:
    enabled: true
    samples: 100
    min_article_length: 500

  gutenberg:
    enabled: true
    samples: 50
    genres: ["Fiction", "Science", "History"]

  code:
    enabled: true
    samples: 200
    languages: ["Python", "JavaScript", "Java", "Go"]

  # ... etc
```

## Output Format

### Documents (`data/raw/`)

Each source produces a JSONL file (e.g., `wikipedia.jsonl`):

```json
{
  "id": "wikipedia_Machine_learning",
  "source": "wikipedia",
  "source_id": "12345",
  "text": "Machine learning is a subset of artificial intelligence...",
  "metadata": {
    "title": "Machine learning",
    "url": "https://en.wikipedia.org/wiki/Machine_learning",
    "word_count": 1500,
    "char_count": 8500,
    "fetch_timestamp": "2024-01-15T10:30:00Z"
  },
  "source_specific": {
    "page_id": "12345",
    "categories": ["Artificial intelligence", "Machine learning"]
  }
}
```

### Chunks (`data/chunks/`)

Pre-chunked snippets (e.g., `wikipedia_chunks.jsonl`):

```json
{
  "id": "wikipedia_Machine_learning_chunk_001",
  "document_id": "wikipedia_Machine_learning",
  "source": "wikipedia",
  "chunk_index": 0,
  "total_chunks": 15,
  "text": "Machine learning is a subset of artificial intelligence...",
  "metadata": {
    "token_count": 512,
    "char_count": 2100,
    "start_char": 0,
    "end_char": 2100
  }
}
```

## Directory Structure

```
DataGathering/
├── src/datagather/          # Source code
│   ├── cli.py               # CLI entry point
│   ├── config.py            # Configuration models
│   ├── sources/             # Data source implementations
│   ├── storage/             # JSONL/Parquet writers
│   └── utils/               # Chunking, rate limiting, etc.
├── data/                    # Output (git-ignored)
│   ├── raw/                 # Full documents
│   ├── chunks/              # Pre-chunked snippets
│   └── checkpoints/         # Resume state
├── config.yaml              # Default configuration
├── pyproject.toml           # Project metadata
└── README.md
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/datagather

# Linting
ruff check src/
```

## Optional Dependencies

Some sources require optional dependencies:

```bash
# For news-please (live Common Crawl scraping)
uv pip install -e ".[news]"

# For ConvoKit (Oyez transcripts)
uv pip install -e ".[conversational]"

# All optional dependencies
uv pip install -e ".[all]"
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project uses data from:
- [Wikipedia](https://www.wikipedia.org/) (Wikimedia Foundation)
- [Project Gutenberg](https://www.gutenberg.org/)
- [arXiv](https://arxiv.org/) (Cornell University)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law)
- Various open-source dialog datasets

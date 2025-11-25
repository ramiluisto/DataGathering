"""Code source for gathering code from HuggingFace datasets."""

import logging
import random
from typing import Iterator, Optional

from datagather.config import CodeConfig
from datagather.sources.base import BaseSource, Document
from datagather.utils.text import clean_text

logger = logging.getLogger(__name__)

# Try to import datasets
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.warning("datasets package not installed")


class CodeSource(BaseSource):
    """Code source using HuggingFace datasets.

    Supports:
    - codeparrot/github-code: 1TB of code from GitHub
    - bigcode/the-stack-v2: 3.1TB of permissively licensed code

    Uses streaming mode to handle large datasets efficiently.
    """

    name = "code"

    def __init__(self, config: CodeConfig, checkpoint_state: Optional[dict] = None):
        """Initialize code source.

        Args:
            config: Code configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: CodeConfig = config

    def initialize(self) -> None:
        """Initialize (nothing to do for HuggingFace datasets)."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _should_include_file(self, sample: dict) -> bool:
        """Check if a file should be included based on config.

        Args:
            sample: Dataset sample

        Returns:
            True if file should be included
        """
        # Check file size
        code = sample.get("code", sample.get("content", ""))
        size = len(code.encode("utf-8"))

        if size < self.config.min_file_size:
            return False
        if size > self.config.max_file_size:
            return False

        # Check exclude patterns
        path = sample.get("path", "")
        for pattern in self.config.exclude_patterns:
            # Simple pattern matching
            if pattern.startswith("**/"):
                if pattern[3:].replace("*", "") in path:
                    return False
            elif pattern.endswith("/**"):
                if path.startswith(pattern[:-3]):
                    return False
            elif "*" in pattern:
                # Basic wildcard matching
                parts = pattern.split("*")
                if all(part in path for part in parts if part):
                    return False

        return True

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch code files from HuggingFace datasets.

        Args:
            limit: Maximum number of files to fetch

        Yields:
            Document objects
        """
        if not HAS_DATASETS:
            raise ImportError("datasets package required")

        # Calculate samples per language
        languages = self.config.languages or ["Python"]
        if self.config.samples_per_language:
            samples_per_lang = self.config.samples_per_language
        else:
            samples_per_lang = (limit // len(languages)) + 1

        fetched = 0
        lang_counts = {lang: 0 for lang in languages}

        for language in languages:
            if fetched >= limit:
                break

            logger.info(f"Loading code dataset for {language}...")

            try:
                # Load dataset in streaming mode
                if "codeparrot" in self.config.dataset:
                    ds = load_dataset(
                        self.config.dataset,
                        languages=[language],
                        streaming=True,
                        split="train",
                        trust_remote_code=True,
                    )
                else:
                    # The Stack or similar
                    ds = load_dataset(
                        self.config.dataset,
                        streaming=True,
                        split="train",
                        trust_remote_code=True,
                    )

                lang_fetched = 0
                for sample in ds:
                    if fetched >= limit or lang_fetched >= samples_per_lang:
                        break

                    # Check language for datasets that include multiple
                    sample_lang = sample.get("language", language)
                    if sample_lang != language:
                        continue

                    # Check inclusion criteria
                    if not self._should_include_file(sample):
                        continue

                    # Get code content
                    code = sample.get("code", sample.get("content", ""))
                    if not code:
                        continue

                    # Generate document ID
                    repo = sample.get("repo_name", sample.get("repository_name", "unknown"))
                    path = sample.get("path", sample.get("filepath", "unknown"))
                    doc_id = f"code_{language}_{repo}_{path}".replace("/", "_").replace(" ", "_")

                    # Skip if already processed
                    if self._is_processed(doc_id):
                        continue

                    # Create document
                    doc = Document(
                        id=doc_id,
                        source="code",
                        source_id=f"{repo}/{path}",
                        text=code,  # Don't clean code - preserve formatting
                        metadata={
                            "title": path.split("/")[-1],
                            "language": language,
                            "repository": repo,
                        },
                        source_specific={
                            "repo_name": repo,
                            "path": path,
                            "language": language,
                            "size_bytes": len(code.encode("utf-8")),
                            "license": sample.get("license", "unknown"),
                        },
                    )

                    self._mark_processed(doc_id)
                    fetched += 1
                    lang_fetched += 1
                    lang_counts[language] = lang_fetched
                    yield doc

            except Exception as e:
                logger.warning(f"Error loading dataset for {language}: {e}")
                continue

        logger.info(f"Code: fetched {fetched} files across languages: {lang_counts}")

    def validate_config(self) -> list[str]:
        """Validate code configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not self.config.languages:
            errors.append("at least one language required")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        if self.config.min_file_size >= self.config.max_file_size:
            errors.append("min_file_size must be less than max_file_size")
        return errors

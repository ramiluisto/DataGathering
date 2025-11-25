"""Reddit source for gathering posts and comments from HuggingFace datasets."""

import logging
import random
from typing import Iterator, Optional

from datagather.config import RedditConfig
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


class RedditSource(BaseSource):
    """Reddit source using HuggingFace datasets.

    Since the Pushshift API is now restricted, this uses archived Reddit data
    available on HuggingFace. Several datasets are available:
    - webis/tldr-17 (summarization dataset with posts)
    - sentence-transformers/reddit-title-body (title/body pairs)

    For historical Pushshift archives, use the archive_path option.
    """

    name = "reddit"

    def __init__(self, config: RedditConfig, checkpoint_state: Optional[dict] = None):
        """Initialize Reddit source.

        Args:
            config: Reddit configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: RedditConfig = config

    def initialize(self) -> None:
        """Initialize resources."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _get_huggingface_data(self, limit: int) -> Iterator[dict]:
        """Load Reddit data from HuggingFace datasets.

        Args:
            limit: Maximum samples to fetch

        Yields:
            Reddit post/comment dictionaries
        """
        # Try several Reddit datasets
        datasets_to_try = [
            ("webis/tldr-17", "train"),
            ("sentence-transformers/reddit-title-body", "train"),
        ]

        for dataset_name, split in datasets_to_try:
            try:
                logger.info(f"Loading Reddit dataset: {dataset_name}")
                ds = load_dataset(
                    dataset_name,
                    streaming=True,
                    split=split,
                    trust_remote_code=True,
                )

                count = 0
                for sample in ds:
                    if count >= limit:
                        return

                    # Normalize different dataset formats
                    if "content" in sample:
                        text = sample["content"]
                        title = sample.get("title", "")
                    elif "body" in sample:
                        text = sample["body"]
                        title = sample.get("title", "")
                    elif "selftext" in sample:
                        text = sample["selftext"]
                        title = sample.get("title", "")
                    else:
                        continue

                    # Filter by subreddit if configured
                    subreddit = sample.get("subreddit", "unknown")
                    if self.config.subreddits:
                        if subreddit.lower() not in [s.lower() for s in self.config.subreddits]:
                            continue

                    # Filter by score if available
                    score = sample.get("score", sample.get("ups", 0))
                    if score < self.config.min_score:
                        continue

                    # Filter by length
                    if len(text) < self.config.min_comment_length:
                        continue

                    yield {
                        "text": text,
                        "title": title,
                        "subreddit": subreddit,
                        "score": score,
                        "id": sample.get("id", str(count)),
                        "author": sample.get("author", "unknown"),
                        "created_utc": sample.get("created_utc"),
                        "is_submission": "title" in sample and sample["title"],
                    }
                    count += 1

                # If we got data from this dataset, don't try others
                if count > 0:
                    return

            except Exception as e:
                logger.warning(f"Error loading {dataset_name}: {e}")
                continue

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch Reddit posts/comments.

        Args:
            limit: Maximum number of items to fetch

        Yields:
            Document objects
        """
        if not HAS_DATASETS:
            raise ImportError("datasets package required")

        fetched = 0

        for item in self._get_huggingface_data(limit * 2):  # Fetch extra to account for filtering
            if fetched >= limit:
                break

            item_id = item.get("id", "")
            subreddit = item.get("subreddit", "unknown")
            doc_id = f"reddit_{subreddit}_{item_id}"

            # Skip if already processed
            if self._is_processed(doc_id):
                continue

            # Combine title and text for submissions
            text = item.get("text", "")
            title = item.get("title", "")
            if title and item.get("is_submission"):
                full_text = f"{title}\n\n{text}" if text else title
            else:
                full_text = text

            if not full_text.strip():
                continue

            full_text = clean_text(full_text)

            doc = Document(
                id=doc_id,
                source="reddit",
                source_id=item_id,
                text=full_text,
                metadata={
                    "title": title,
                    "subreddit": subreddit,
                    "score": item.get("score", 0),
                },
                source_specific={
                    "post_id": item_id,
                    "subreddit": subreddit,
                    "author": item.get("author", "unknown"),
                    "score": item.get("score", 0),
                    "created_utc": item.get("created_utc"),
                    "is_submission": item.get("is_submission", True),
                },
            )

            self._mark_processed(doc_id)
            fetched += 1
            yield doc

        logger.info(f"Reddit: fetched {fetched} posts/comments")

    def validate_config(self) -> list[str]:
        """Validate Reddit configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

"""News source for gathering articles from HuggingFace datasets."""

import logging
from typing import Iterator, Optional

from datagather.config import NewsConfig
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


class NewsSource(BaseSource):
    """News source using HuggingFace datasets.

    Supports several news datasets:
    - cc_news: Common Crawl news dataset
    - multi_news: Multi-document news summarization
    - cnn_dailymail: CNN/DailyMail news articles

    For live Common Crawl scraping, use the news-please package
    (requires optional dependency: uv pip install 'datagather[news]')
    """

    name = "news"

    def __init__(self, config: NewsConfig, checkpoint_state: Optional[dict] = None):
        """Initialize news source.

        Args:
            config: News configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: NewsConfig = config

    def initialize(self) -> None:
        """Initialize resources."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _load_cc_news(self, limit: int) -> Iterator[dict]:
        """Load CC-News dataset.

        Args:
            limit: Maximum samples

        Yields:
            News article dictionaries
        """
        try:
            ds = load_dataset(
                "cc_news",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                text = sample.get("text", "")
                if len(text) < self.config.min_article_length:
                    continue

                yield {
                    "text": text,
                    "title": sample.get("title", ""),
                    "url": sample.get("url", ""),
                    "domain": sample.get("domain", "unknown"),
                    "date": sample.get("date", ""),
                    "authors": sample.get("authors", []),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading cc_news: {e}")

    def _load_cnn_dailymail(self, limit: int) -> Iterator[dict]:
        """Load CNN/DailyMail dataset.

        Args:
            limit: Maximum samples

        Yields:
            News article dictionaries
        """
        try:
            ds = load_dataset(
                "cnn_dailymail",
                "3.0.0",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                text = sample.get("article", "")
                if len(text) < self.config.min_article_length:
                    continue

                yield {
                    "text": text,
                    "title": "",  # CNN/DailyMail doesn't have separate titles
                    "url": sample.get("id", ""),
                    "domain": "cnn.com" if "cnn" in sample.get("id", "").lower() else "dailymail.co.uk",
                    "date": "",
                    "authors": [],
                    "highlights": sample.get("highlights", ""),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading cnn_dailymail: {e}")

    def _load_multi_news(self, limit: int) -> Iterator[dict]:
        """Load Multi-News dataset.

        Args:
            limit: Maximum samples

        Yields:
            News article dictionaries
        """
        try:
            ds = load_dataset(
                "multi_news",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                # Multi-news has multiple source documents
                text = sample.get("document", "")
                if len(text) < self.config.min_article_length:
                    continue

                yield {
                    "text": text,
                    "title": "",
                    "url": "",
                    "domain": "multi_news",
                    "date": "",
                    "authors": [],
                    "summary": sample.get("summary", ""),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading multi_news: {e}")

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch news articles.

        Args:
            limit: Maximum number of articles to fetch

        Yields:
            Document objects
        """
        if not HAS_DATASETS:
            raise ImportError("datasets package required")

        fetched = 0

        # Try different news datasets
        data_generators = [
            self._load_cc_news(limit),
            self._load_cnn_dailymail(limit),
            self._load_multi_news(limit),
        ]

        for gen in data_generators:
            if fetched >= limit:
                break

            for article in gen:
                if fetched >= limit:
                    break

                # Generate document ID
                url = article.get("url", "")
                domain = article.get("domain", "unknown")
                doc_id = f"news_{domain}_{hash(url + article.get('text', '')[:100])}"

                # Skip if already processed
                if self._is_processed(doc_id):
                    continue

                # Build text with optional title
                text = article.get("text", "")
                title = article.get("title", "")

                if self.config.include_title and title:
                    full_text = f"{title}\n\n{text}"
                else:
                    full_text = text

                full_text = clean_text(full_text)

                # Build metadata
                metadata = {
                    "title": title,
                    "url": url,
                    "source_domain": domain,
                }
                if self.config.include_authors:
                    metadata["authors"] = article.get("authors", [])

                doc = Document(
                    id=doc_id,
                    source="news",
                    source_id=url or str(hash(text[:100])),
                    text=full_text,
                    metadata=metadata,
                    source_specific={
                        "domain": domain,
                        "date": article.get("date", ""),
                        "highlights": article.get("highlights", ""),
                        "summary": article.get("summary", ""),
                    },
                )

                self._mark_processed(doc_id)
                fetched += 1
                yield doc

        logger.info(f"News: fetched {fetched} articles")

    def validate_config(self) -> list[str]:
        """Validate news configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

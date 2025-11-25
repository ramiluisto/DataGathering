"""Twitter source for gathering tweets from archived datasets."""

import logging
from typing import Iterator, Optional

from datagather.config import TwitterConfig
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


class TwitterSource(BaseSource):
    """Twitter source using archived datasets.

    Since the Twitter API is now paid, this uses archived Twitter data
    available on HuggingFace or from GESIS archives.

    Available datasets:
    - sentiment140: 1.6M tweets with sentiment labels
    - tweet_eval: Various Twitter NLP tasks
    - covid_tweets: COVID-19 related tweets

    Note: Many Twitter datasets only contain tweet IDs due to Twitter's TOS.
    This source uses datasets that include full text.
    """

    name = "twitter"

    def __init__(self, config: TwitterConfig, checkpoint_state: Optional[dict] = None):
        """Initialize Twitter source.

        Args:
            config: Twitter configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: TwitterConfig = config

    def initialize(self) -> None:
        """Initialize resources."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _load_sentiment140(self, limit: int) -> Iterator[dict]:
        """Load Sentiment140 dataset.

        Args:
            limit: Maximum samples

        Yields:
            Tweet dictionaries
        """
        try:
            ds = load_dataset(
                "sentiment140",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                text = sample.get("text", "")

                # Filter by minimum length
                if len(text) < self.config.min_tweet_length:
                    continue

                # Filter retweets if configured
                if self.config.exclude_retweets and text.startswith("RT "):
                    continue

                yield {
                    "text": text,
                    "dataset": "sentiment140",
                    "sentiment": sample.get("sentiment", 0),
                    "user": sample.get("user", ""),
                    "date": sample.get("date", ""),
                    "query": sample.get("query", ""),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading sentiment140: {e}")

    def _load_tweet_eval(self, limit: int) -> Iterator[dict]:
        """Load TweetEval dataset.

        Args:
            limit: Maximum samples

        Yields:
            Tweet dictionaries
        """
        try:
            # TweetEval has multiple tasks, use sentiment
            ds = load_dataset(
                "tweet_eval",
                "sentiment",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                text = sample.get("text", "")

                # Filter by minimum length
                if len(text) < self.config.min_tweet_length:
                    continue

                # Filter retweets if configured
                if self.config.exclude_retweets and text.startswith("RT "):
                    continue

                yield {
                    "text": text,
                    "dataset": "tweet_eval",
                    "label": sample.get("label", 0),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading tweet_eval: {e}")

    def _load_hate_speech(self, limit: int) -> Iterator[dict]:
        """Load hate speech tweets dataset.

        Args:
            limit: Maximum samples

        Yields:
            Tweet dictionaries (non-hateful tweets only for diversity)
        """
        try:
            ds = load_dataset(
                "tweets_hate_speech_detection",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                # Only include non-hateful tweets (label 0)
                if sample.get("label", 1) != 0:
                    continue

                text = sample.get("tweet", "")

                # Filter by minimum length
                if len(text) < self.config.min_tweet_length:
                    continue

                # Filter retweets if configured
                if self.config.exclude_retweets and text.startswith("RT "):
                    continue

                yield {
                    "text": text,
                    "dataset": "tweets_hate_speech",
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading tweets_hate_speech: {e}")

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch tweets from archived datasets.

        Args:
            limit: Maximum number of tweets to fetch

        Yields:
            Document objects
        """
        if not HAS_DATASETS:
            raise ImportError("datasets package required")

        fetched = 0

        # Try different tweet datasets
        loaders = [
            self._load_sentiment140,
            self._load_tweet_eval,
            self._load_hate_speech,
        ]

        samples_per_loader = limit // len(loaders)

        for loader in loaders:
            if fetched >= limit:
                break

            for tweet_data in loader(samples_per_loader):
                if fetched >= limit:
                    break

                text = tweet_data.get("text", "")
                dataset = tweet_data.get("dataset", "unknown")
                doc_id = f"twitter_{dataset}_{hash(text)}"

                # Skip if already processed
                if self._is_processed(doc_id):
                    continue

                # Clean text (but preserve Twitter-specific formatting)
                text = clean_text(text)

                doc = Document(
                    id=doc_id,
                    source="twitter",
                    source_id=str(hash(text)),
                    text=text,
                    metadata={
                        "title": "",
                        "dataset": dataset,
                    },
                    source_specific={
                        "dataset": dataset,
                        "sentiment": tweet_data.get("sentiment", tweet_data.get("label")),
                        "user": tweet_data.get("user", ""),
                        "date": tweet_data.get("date", ""),
                    },
                )

                self._mark_processed(doc_id)
                fetched += 1
                yield doc

        logger.info(f"Twitter: fetched {fetched} tweets")

    def validate_config(self) -> list[str]:
        """Validate Twitter configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

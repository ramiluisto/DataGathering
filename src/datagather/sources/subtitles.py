"""Subtitles source for gathering movie/TV subtitles and transcripts."""

import logging
import re
from typing import Iterator, Optional

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


def clean_subtitle_text(text: str) -> str:
    """Clean subtitle text by removing timing markers and formatting.

    Args:
        text: Raw subtitle text

    Returns:
        Cleaned text
    """
    # Remove subtitle timing markers (e.g., "00:01:23,456 --> 00:01:25,789")
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", text)

    # Remove sequence numbers
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove HTML-style tags (like <i>, </i>, <b>, etc.)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove speaker indicators like "JOHN:" or "[John]"
    text = re.sub(r"^\[[^\]]+\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[A-Z][A-Z\s]+:\s*", "", text, flags=re.MULTILINE)

    # Remove sound descriptions like "[music playing]" or "(door closes)"
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)

    # Remove excess whitespace
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


class SubtitlesConfig:
    """Subtitles source configuration."""

    def __init__(
        self,
        enabled: bool = True,
        samples: int = 100,
        dataset: str = "open_subtitles",
        language: str = "en",
        min_length: int = 500,
        combine_utterances: bool = True,
        max_utterances_per_doc: int = 50,
    ):
        self.enabled = enabled
        self.samples = samples
        self.dataset = dataset
        self.language = language
        self.min_length = min_length
        self.combine_utterances = combine_utterances
        self.max_utterances_per_doc = max_utterances_per_doc


class SubtitlesSource(BaseSource):
    """Subtitles source for gathering movie/TV subtitles.

    Uses HuggingFace datasets for OpenSubtitles data.
    Provides informal spoken English with dialogue patterns.
    """

    name = "subtitles"

    def __init__(self, config, checkpoint_state: Optional[dict] = None):
        """Initialize Subtitles source.

        Args:
            config: Subtitles configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config = config
        self._dataset = None
        self._iterator = None
        self._buffer = []  # Buffer for combining utterances

    def initialize(self) -> None:
        """Initialize dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

        try:
            logger.info(f"Loading subtitles dataset: {self.config.dataset}")

            # OpenSubtitles is a parallel corpus, we just use the English side
            if self.config.dataset == "open_subtitles":
                # Load OpenSubtitles English-English (monolingual extraction)
                self._dataset = load_dataset(
                    "open_subtitles",
                    lang1="en",
                    lang2="en",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
            else:
                self._dataset = load_dataset(
                    self.config.dataset,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )

            self._iterator = iter(self._dataset)

        except Exception as e:
            logger.warning(f"Failed to load OpenSubtitles: {e}")
            # Try alternative TED talks dataset
            logger.info("Falling back to TED talks dataset")
            try:
                self._dataset = load_dataset(
                    "ted_talks_iwslt",
                    language_pair=("en", "en"),
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                self._iterator = iter(self._dataset)
            except Exception as e2:
                logger.warning(f"Failed to load TED talks: {e2}")
                # Final fallback: movie dialogs
                logger.info("Falling back to movie dialog dataset")
                self._dataset = load_dataset(
                    "cornell_movie_dialog",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                self._iterator = iter(self._dataset)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._dataset = None
        self._iterator = None
        self._buffer = []

    def _get_next_document(self) -> Optional[tuple[str, dict]]:
        """Get the next document by combining utterances.

        Returns:
            Tuple of (text, metadata) or None if no more data
        """
        utterances = []
        movie_id = None

        # Collect utterances until we have enough
        while len(utterances) < self.config.max_utterances_per_doc:
            try:
                item = next(self._iterator)
            except StopIteration:
                break

            # Handle OpenSubtitles format
            if "translation" in item:
                trans = item["translation"]
                if "en" in trans:
                    text = trans["en"]
                    current_id = item.get("id", "unknown")
                else:
                    continue
            # Handle movie dialog format
            elif "utterance" in item:
                text = item["utterance"].get("text", "")
                current_id = item["utterance"].get("movie_id", "unknown")
            # Handle TED talks format
            elif "talk_id" in item:
                text = item.get("text", "")
                current_id = str(item.get("talk_id", "unknown"))
            else:
                # Generic format
                text = item.get("text", item.get("sentence", ""))
                current_id = item.get("id", "unknown")

            if not text:
                continue

            # If we're combining by movie/talk, check if we switched
            if movie_id is not None and current_id != movie_id and utterances:
                # Put this item back (can't actually do this with iterator, so just flush)
                break

            movie_id = current_id
            utterances.append(text)

        if not utterances:
            return None

        # Combine utterances into a document
        full_text = "\n".join(utterances)
        full_text = clean_subtitle_text(full_text)
        full_text = clean_text(full_text)

        if len(full_text) < self.config.min_length:
            return None

        metadata = {
            "movie_id": movie_id,
            "utterance_count": len(utterances),
        }

        return full_text, metadata

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch subtitle documents.

        Args:
            limit: Maximum number of documents to fetch

        Yields:
            Document objects
        """
        if not self._iterator:
            self.initialize()

        fetched = 0
        attempts = 0
        max_attempts = limit * 3

        while fetched < limit and attempts < max_attempts:
            attempts += 1

            result = self._get_next_document()
            if result is None:
                logger.info("No more subtitle data available")
                break

            text, metadata = result
            movie_id = metadata.get("movie_id", str(attempts))
            doc_id = f"subtitles_{movie_id}_{fetched}"

            if self._is_processed(doc_id):
                continue

            doc = Document(
                id=doc_id,
                source="subtitles",
                source_id=f"{movie_id}_{fetched}",
                text=text,
                metadata={
                    "type": "dialogue",
                    "movie_id": movie_id,
                },
                source_specific=metadata,
            )

            self._mark_processed(doc_id)
            fetched += 1
            yield doc

        logger.info(f"Subtitles: fetched {fetched} documents")

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if self.config.min_length < 0:
            errors.append("min_length must be non-negative")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

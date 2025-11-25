"""Stack Overflow source for gathering Q&A pairs."""

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


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Text with potential HTML tags

    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Fix multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class StackOverflowConfig:
    """Stack Overflow source configuration."""

    def __init__(
        self,
        enabled: bool = True,
        samples: int = 100,
        dataset: str = "koutch/stackoverflow_python",
        min_score: int = 5,
        min_answer_score: int = 3,
        include_code: bool = True,
        languages: Optional[list[str]] = None,
    ):
        self.enabled = enabled
        self.samples = samples
        self.dataset = dataset
        self.min_score = min_score
        self.min_answer_score = min_answer_score
        self.include_code = include_code
        self.languages = languages or ["python"]


class StackOverflowSource(BaseSource):
    """Stack Overflow source for gathering Q&A data.

    Uses HuggingFace datasets for Stack Overflow archives.
    Formats questions with their accepted/top answers.
    """

    name = "stackoverflow"

    # Available datasets for different programming languages
    DATASETS = {
        "python": "koutch/stackoverflow_python",
        "javascript": "koutch/stackoverflow_javascript",
        "java": "koutch/stackoverflow_java",
        # Generic Stack Exchange dataset
        "general": "HuggingFaceH4/stack-exchange-preferences",
    }

    def __init__(self, config, checkpoint_state: Optional[dict] = None):
        """Initialize Stack Overflow source.

        Args:
            config: Stack Overflow configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config = config
        self._dataset = None
        self._iterator = None

    def initialize(self) -> None:
        """Initialize dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

        # Determine which dataset to use
        dataset_name = self.config.dataset

        try:
            logger.info(f"Loading Stack Overflow dataset: {dataset_name}")
            self._dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self._iterator = iter(self._dataset)
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            # Fall back to Stack Exchange preferences dataset
            logger.info("Falling back to stack-exchange-preferences dataset")
            self._dataset = load_dataset(
                "HuggingFaceH4/stack-exchange-preferences",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self._iterator = iter(self._dataset)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._dataset = None
        self._iterator = None

    def _format_qa(self, item: dict) -> Optional[tuple[str, dict]]:
        """Format a Q&A item into text.

        Args:
            item: Dataset item

        Returns:
            Tuple of (formatted_text, metadata) or None if invalid
        """
        # Handle different dataset formats
        if "question" in item and "answer" in item:
            # koutch/stackoverflow_* format
            question = item.get("question", "")
            answer = item.get("answer", "")
            title = item.get("title", "")
            score = item.get("score", 0)
            answer_score = item.get("answer_score", 0)

            if score < self.config.min_score:
                return None

            # Strip HTML if present
            question = strip_html_tags(question)
            answer = strip_html_tags(answer)
            title = strip_html_tags(title)

            text = f"Question: {title}\n\n{question}\n\nAnswer:\n{answer}"

            metadata = {
                "title": title,
                "score": score,
                "answer_score": answer_score,
                "tags": item.get("tags", []),
            }

        elif "question_id" in item:
            # Stack Exchange preferences format
            question = item.get("question", "")
            answers = item.get("answers", [])

            if not answers:
                return None

            # Get best answer
            best_answer = max(answers, key=lambda x: x.get("pm_score", 0))

            question = strip_html_tags(question)
            answer_text = strip_html_tags(best_answer.get("text", ""))

            text = f"Question:\n{question}\n\nAnswer:\n{answer_text}"

            metadata = {
                "question_id": item.get("question_id"),
                "answer_score": best_answer.get("pm_score", 0),
            }

        else:
            return None

        # Clean the text
        text = clean_text(text)

        if len(text) < 100:  # Skip very short Q&As
            return None

        return text, metadata

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch Q&A pairs from Stack Overflow.

        Args:
            limit: Maximum number of Q&As to fetch

        Yields:
            Document objects
        """
        if not self._iterator:
            self.initialize()

        fetched = 0
        attempts = 0
        max_attempts = limit * 5  # Allow for filtering

        while fetched < limit and attempts < max_attempts:
            try:
                item = next(self._iterator)
                attempts += 1
            except StopIteration:
                logger.info("Reached end of Stack Overflow dataset")
                break

            # Create document ID
            if "question_id" in item:
                q_id = str(item["question_id"])
            elif "id" in item:
                q_id = str(item["id"])
            else:
                q_id = str(attempts)

            doc_id = f"stackoverflow_{q_id}"

            if self._is_processed(doc_id):
                continue

            result = self._format_qa(item)
            if result is None:
                continue

            text, metadata = result

            doc = Document(
                id=doc_id,
                source="stackoverflow",
                source_id=q_id,
                text=text,
                metadata={
                    "title": metadata.get("title", ""),
                    "url": f"https://stackoverflow.com/q/{q_id}",
                },
                source_specific=metadata,
            )

            self._mark_processed(doc_id)
            fetched += 1
            yield doc

        logger.info(f"Stack Overflow: fetched {fetched} Q&As in {attempts} attempts")

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

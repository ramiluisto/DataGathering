"""Conversational source for gathering dialog and transcript data."""

import logging
from typing import Iterator, Optional

from datagather.config import ConversationalConfig
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


class ConversationalSource(BaseSource):
    """Conversational source for dialog and transcript data.

    Supports:
    - DailyDialog: Multi-turn dialogues
    - Topical Chat: Open-domain conversations
    - Oyez Supreme Court transcripts (via ConvoKit or direct)

    For Oyez transcripts, requires ConvoKit:
        uv pip install convokit
    """

    name = "conversational"

    def __init__(self, config: ConversationalConfig, checkpoint_state: Optional[dict] = None):
        """Initialize conversational source.

        Args:
            config: Conversational configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: ConversationalConfig = config

    def initialize(self) -> None:
        """Initialize resources."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _load_daily_dialog(self, limit: int) -> Iterator[dict]:
        """Load DailyDialog dataset.

        Args:
            limit: Maximum samples

        Yields:
            Dialog dictionaries
        """
        try:
            ds = load_dataset(
                "daily_dialog",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                dialog = sample.get("dialog", [])
                if not dialog:
                    continue

                # Join dialog turns
                text = "\n".join([f"- {turn}" for turn in dialog])

                yield {
                    "text": text,
                    "dataset": "daily_dialog",
                    "num_turns": len(dialog),
                    "act": sample.get("act", []),
                    "emotion": sample.get("emotion", []),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading daily_dialog: {e}")

    def _load_topical_chat(self, limit: int) -> Iterator[dict]:
        """Load Topical Chat dataset.

        Args:
            limit: Maximum samples

        Yields:
            Dialog dictionaries
        """
        try:
            ds = load_dataset(
                "alexa/Topical-Chat",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                # Extract conversation
                conversation = sample.get("content", [])
                if not conversation:
                    # Try alternative format
                    message = sample.get("message", "")
                    if message:
                        conversation = [message]

                if not conversation:
                    continue

                # Join conversation turns
                if isinstance(conversation, list):
                    text = "\n".join([f"- {turn}" for turn in conversation if turn])
                else:
                    text = str(conversation)

                yield {
                    "text": text,
                    "dataset": "topical_chat",
                    "num_turns": len(conversation) if isinstance(conversation, list) else 1,
                    "sentiment": sample.get("sentiment", ""),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading topical_chat: {e}")

    def _load_empathetic_dialogues(self, limit: int) -> Iterator[dict]:
        """Load Empathetic Dialogues dataset.

        Args:
            limit: Maximum samples

        Yields:
            Dialog dictionaries
        """
        try:
            ds = load_dataset(
                "empathetic_dialogues",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            current_conv = []
            current_conv_id = None

            for sample in ds:
                if count >= limit:
                    return

                conv_id = sample.get("conv_id", "")
                utterance = sample.get("utterance", "")

                if conv_id != current_conv_id and current_conv:
                    # Yield previous conversation
                    text = "\n".join([f"- {turn}" for turn in current_conv])
                    yield {
                        "text": text,
                        "dataset": "empathetic_dialogues",
                        "num_turns": len(current_conv),
                        "emotion": sample.get("context", ""),
                    }
                    count += 1
                    current_conv = []

                current_conv_id = conv_id
                if utterance:
                    current_conv.append(utterance)

            # Don't forget last conversation
            if current_conv and count < limit:
                text = "\n".join([f"- {turn}" for turn in current_conv])
                yield {
                    "text": text,
                    "dataset": "empathetic_dialogues",
                    "num_turns": len(current_conv),
                }

        except Exception as e:
            logger.warning(f"Error loading empathetic_dialogues: {e}")

    def _load_persona_chat(self, limit: int) -> Iterator[dict]:
        """Load Persona-Chat dataset.

        Args:
            limit: Maximum samples

        Yields:
            Dialog dictionaries
        """
        try:
            ds = load_dataset(
                "bavard/personachat_truecased",
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                history = sample.get("history", [])
                if not history:
                    continue

                text = "\n".join([f"- {turn}" for turn in history])

                yield {
                    "text": text,
                    "dataset": "persona_chat",
                    "num_turns": len(history),
                    "personas": sample.get("personality", []),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading persona_chat: {e}")

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch conversational data.

        Args:
            limit: Maximum number of conversations to fetch

        Yields:
            Document objects
        """
        if not HAS_DATASETS:
            raise ImportError("datasets package required")

        fetched = 0

        # Get dialog config
        dialog_config = self.config.datasets.dialog
        if dialog_config.enabled:
            samples_per_source = dialog_config.samples // len(dialog_config.sources)

            # Map source names to loaders
            loaders = {
                "daily_dialog": self._load_daily_dialog,
                "topical_chat": self._load_topical_chat,
                "empathetic_dialogues": self._load_empathetic_dialogues,
                "persona_chat": self._load_persona_chat,
            }

            for source_name in dialog_config.sources:
                if fetched >= limit:
                    break

                loader = loaders.get(source_name)
                if not loader:
                    logger.warning(f"Unknown dialog source: {source_name}")
                    continue

                for dialog_data in loader(samples_per_source):
                    if fetched >= limit:
                        break

                    text = dialog_data.get("text", "")
                    dataset = dialog_data.get("dataset", source_name)
                    doc_id = f"conversational_{dataset}_{hash(text[:100])}"

                    # Skip if already processed
                    if self._is_processed(doc_id):
                        continue

                    # Clean text
                    text = clean_text(text)

                    doc = Document(
                        id=doc_id,
                        source="conversational",
                        source_id=str(hash(text[:100])),
                        text=text,
                        metadata={
                            "title": f"{dataset} conversation",
                            "dataset": dataset,
                            "num_turns": dialog_data.get("num_turns", 0),
                        },
                        source_specific={
                            "dataset": dataset,
                            "num_turns": dialog_data.get("num_turns", 0),
                            "emotion": dialog_data.get("emotion", ""),
                            "sentiment": dialog_data.get("sentiment", ""),
                        },
                    )

                    self._mark_processed(doc_id)
                    fetched += 1
                    yield doc

        logger.info(f"Conversational: fetched {fetched} conversations")

    def validate_config(self) -> list[str]:
        """Validate conversational configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

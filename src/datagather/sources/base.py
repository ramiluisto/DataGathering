"""Abstract base class for all data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, Optional


@dataclass
class Document:
    """Standard document format for all sources.

    Attributes:
        id: Unique identifier for the document (source_sourceId format)
        source: Name of the data source (e.g., "wikipedia", "gutenberg")
        source_id: Original ID from the source
        text: Full document text
        metadata: Common metadata (title, url, word_count, etc.)
        source_specific: Source-specific metadata
    """

    id: str
    source: str
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_specific: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "text": self.text,
            "metadata": {
                **self.metadata,
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "char_count": len(self.text),
                "word_count": len(self.text.split()),
            },
            "source_specific": self.source_specific,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(
            id=data["id"],
            source=data["source"],
            source_id=data["source_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            source_specific=data.get("source_specific", {}),
        )


@dataclass
class Chunk:
    """A chunk of a document.

    Attributes:
        id: Unique identifier (document_id_chunk_index format)
        document_id: ID of the parent document
        source: Name of the data source
        chunk_index: Index of this chunk in the document
        total_chunks: Total number of chunks in the document
        text: Chunk text
        metadata: Chunk metadata (token_count, char positions, etc.)
    """

    id: str
    document_id: str
    source: str
    chunk_index: int
    total_chunks: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            source=data["source"],
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


class BaseSource(ABC):
    """Abstract base class for all data sources.

    All data sources must implement:
    - name: class attribute with source name
    - fetch_documents(): generator yielding Document objects
    - get_checkpoint_state(): return current state for checkpointing

    Optional methods:
    - validate_config(): validate source-specific configuration
    - initialize(): async initialization (API clients, etc.)
    - cleanup(): cleanup resources
    """

    name: str = "base"

    def __init__(self, config: Any, checkpoint_state: Optional[dict] = None):
        """Initialize the source.

        Args:
            config: Source-specific configuration object
            checkpoint_state: Optional checkpoint state to resume from
        """
        self.config = config
        self._checkpoint_state = checkpoint_state or {}
        self._processed_count = self._checkpoint_state.get("completed_samples", 0)
        self._seen_ids: set[str] = set(self._checkpoint_state.get("seen_ids", []))

    @abstractmethod
    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch documents from the source.

        Args:
            limit: Maximum number of documents to fetch

        Yields:
            Document objects
        """
        pass

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return current state for checkpointing.

        Returns:
            Dictionary with checkpoint state
        """
        return {
            "source": self.name,
            "completed_samples": self._processed_count,
            "seen_ids": list(self._seen_ids)[-1000],  # Keep last 1000 IDs
        }

    def validate_config(self) -> list[str]:
        """Validate source-specific configuration.

        Returns:
            List of error messages, empty if valid
        """
        return []

    def initialize(self) -> None:
        """Initialize the source (API clients, downloads, etc.)."""
        pass

    def cleanup(self) -> None:
        """Cleanup resources (close connections, etc.)."""
        pass

    def _mark_processed(self, doc_id: str) -> None:
        """Mark a document as processed."""
        self._processed_count += 1
        self._seen_ids.add(doc_id)

    def _is_processed(self, doc_id: str) -> bool:
        """Check if a document has been processed."""
        return doc_id in self._seen_ids

    @property
    def processed_count(self) -> int:
        """Return number of processed documents."""
        return self._processed_count


class AsyncBaseSource(ABC):
    """Async version of BaseSource for sources that benefit from async I/O."""

    name: str = "async_base"

    def __init__(self, config: Any, checkpoint_state: Optional[dict] = None):
        """Initialize the source."""
        self.config = config
        self._checkpoint_state = checkpoint_state or {}
        self._processed_count = self._checkpoint_state.get("completed_samples", 0)
        self._seen_ids: set[str] = set(self._checkpoint_state.get("seen_ids", []))

    @abstractmethod
    async def fetch_documents(self, limit: int) -> AsyncIterator[Document]:
        """Fetch documents asynchronously.

        Args:
            limit: Maximum number of documents to fetch

        Yields:
            Document objects
        """
        pass

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return current state for checkpointing."""
        return {
            "source": self.name,
            "completed_samples": self._processed_count,
            "seen_ids": list(self._seen_ids)[-1000],
        }

    def validate_config(self) -> list[str]:
        """Validate source-specific configuration."""
        return []

    async def initialize(self) -> None:
        """Async initialization."""
        pass

    async def cleanup(self) -> None:
        """Async cleanup."""
        pass

    def _mark_processed(self, doc_id: str) -> None:
        """Mark a document as processed."""
        self._processed_count += 1
        self._seen_ids.add(doc_id)

    def _is_processed(self, doc_id: str) -> bool:
        """Check if a document has been processed."""
        return doc_id in self._seen_ids

    @property
    def processed_count(self) -> int:
        """Return number of processed documents."""
        return self._processed_count

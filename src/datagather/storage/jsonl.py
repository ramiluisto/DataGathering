"""JSONL streaming writer for documents and chunks."""

from pathlib import Path
from typing import Any, Union

import orjson

from datagather.sources.base import Chunk, Document


class JSONLWriter:
    """Streaming JSONL writer that appends documents/chunks to files.

    This writer is designed for:
    - Memory efficiency: writes one document at a time
    - Resume support: appends to existing files
    - Fast serialization: uses orjson for speed

    Example:
        writer = JSONLWriter(output_dir=Path("./data"))
        writer.write_document(doc)  # Writes to ./data/raw/wikipedia.jsonl
        writer.write_chunk(chunk)   # Writes to ./data/chunks/wikipedia_chunks.jsonl
    """

    def __init__(self, output_dir: Path):
        """Initialize the JSONL writer.

        Args:
            output_dir: Base output directory (will create raw/ and chunks/ subdirs)
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.chunks_dir = self.output_dir / "chunks"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Keep file handles open for performance
        self._doc_handles: dict[str, Any] = {}
        self._chunk_handles: dict[str, Any] = {}

    def _get_doc_handle(self, source: str):
        """Get or create file handle for document source."""
        if source not in self._doc_handles:
            filepath = self.raw_dir / f"{source}.jsonl"
            self._doc_handles[source] = open(filepath, "ab")
        return self._doc_handles[source]

    def _get_chunk_handle(self, source: str):
        """Get or create file handle for chunk source."""
        if source not in self._chunk_handles:
            filepath = self.chunks_dir / f"{source}_chunks.jsonl"
            self._chunk_handles[source] = open(filepath, "ab")
        return self._chunk_handles[source]

    def write_document(self, doc: Union[Document, dict[str, Any]]) -> None:
        """Write a document to the appropriate JSONL file.

        Args:
            doc: Document object or dictionary
        """
        if isinstance(doc, Document):
            data = doc.to_dict()
            source = doc.source
        else:
            data = doc
            source = doc["source"]

        handle = self._get_doc_handle(source)
        handle.write(orjson.dumps(data) + b"\n")
        handle.flush()

    def write_chunk(self, chunk: Union[Chunk, dict[str, Any]]) -> None:
        """Write a chunk to the appropriate JSONL file.

        Args:
            chunk: Chunk object or dictionary
        """
        if isinstance(chunk, Chunk):
            data = chunk.to_dict()
            source = chunk.source
        else:
            data = chunk
            source = chunk["source"]

        handle = self._get_chunk_handle(source)
        handle.write(orjson.dumps(data) + b"\n")
        handle.flush()

    def write_documents(self, docs: list[Union[Document, dict[str, Any]]]) -> int:
        """Write multiple documents.

        Args:
            docs: List of Document objects or dictionaries

        Returns:
            Number of documents written
        """
        for doc in docs:
            self.write_document(doc)
        return len(docs)

    def write_chunks(self, chunks: list[Union[Chunk, dict[str, Any]]]) -> int:
        """Write multiple chunks.

        Args:
            chunks: List of Chunk objects or dictionaries

        Returns:
            Number of chunks written
        """
        for chunk in chunks:
            self.write_chunk(chunk)
        return len(chunks)

    def close(self) -> None:
        """Close all file handles."""
        for handle in self._doc_handles.values():
            handle.close()
        for handle in self._chunk_handles.values():
            handle.close()
        self._doc_handles.clear()
        self._chunk_handles.clear()

    def __enter__(self) -> "JSONLWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def get_document_count(self, source: str) -> int:
        """Count documents in a source file.

        Args:
            source: Source name

        Returns:
            Number of documents in the file
        """
        filepath = self.raw_dir / f"{source}.jsonl"
        if not filepath.exists():
            return 0
        with open(filepath, "rb") as f:
            return sum(1 for _ in f)

    def get_chunk_count(self, source: str) -> int:
        """Count chunks in a source file.

        Args:
            source: Source name

        Returns:
            Number of chunks in the file
        """
        filepath = self.chunks_dir / f"{source}_chunks.jsonl"
        if not filepath.exists():
            return 0
        with open(filepath, "rb") as f:
            return sum(1 for _ in f)


def read_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries
    """
    if not filepath.exists():
        return []
    records = []
    with open(filepath, "rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    return records


def iter_jsonl(filepath: Path):
    """Iterate over records in a JSONL file.

    Args:
        filepath: Path to JSONL file

    Yields:
        Dictionaries from the file
    """
    if not filepath.exists():
        return
    with open(filepath, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)

"""Async orchestrator for parallel source execution."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional, Union

from datagather.sources.base import AsyncBaseSource, BaseSource, Document

logger = logging.getLogger(__name__)


@dataclass
class SourceProgress:
    """Progress tracking for a single source."""

    source_name: str
    target_samples: int
    completed_samples: int = 0
    error_count: int = 0
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_error: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        """Return progress as percentage."""
        if self.target_samples == 0:
            return 100.0
        return (self.completed_samples / self.target_samples) * 100


@dataclass
class OrchestratorProgress:
    """Overall orchestrator progress."""

    sources: dict[str, SourceProgress] = field(default_factory=dict)
    total_documents: int = 0
    started_at: Optional[datetime] = None

    def add_source(self, name: str, target: int) -> None:
        """Add a source to track."""
        self.sources[name] = SourceProgress(source_name=name, target_samples=target)

    def mark_started(self, name: str) -> None:
        """Mark a source as started."""
        if name in self.sources:
            self.sources[name].status = "running"
            self.sources[name].started_at = datetime.now(timezone.utc)

    def mark_completed(self, name: str) -> None:
        """Mark a source as completed."""
        if name in self.sources:
            self.sources[name].status = "completed"
            self.sources[name].completed_at = datetime.now(timezone.utc)

    def mark_failed(self, name: str, error: str) -> None:
        """Mark a source as failed."""
        if name in self.sources:
            self.sources[name].status = "failed"
            self.sources[name].last_error = error
            self.sources[name].completed_at = datetime.now(timezone.utc)

    def increment(self, name: str) -> None:
        """Increment completed count for a source."""
        if name in self.sources:
            self.sources[name].completed_samples += 1
            self.total_documents += 1

    def record_error(self, name: str, error: str) -> None:
        """Record an error for a source."""
        if name in self.sources:
            self.sources[name].error_count += 1
            self.sources[name].last_error = error


class AsyncSourceOrchestrator:
    """Orchestrates parallel execution of multiple data sources.

    Each source runs independently with its own rate limiting,
    allowing maximum throughput without sources blocking each other.

    Example:
        orchestrator = AsyncSourceOrchestrator()

        async for doc in orchestrator.gather_all(sources, limits):
            process(doc)
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[OrchestratorProgress], None]] = None,
        max_concurrent_sources: Optional[int] = None,
    ):
        """Initialize orchestrator.

        Args:
            progress_callback: Optional callback for progress updates
            max_concurrent_sources: Max sources to run at once (None = all)
        """
        self.progress_callback = progress_callback
        self.max_concurrent_sources = max_concurrent_sources
        self.progress = OrchestratorProgress()
        self._document_queue: asyncio.Queue[Optional[Document]] = asyncio.Queue()
        self._active_sources = 0
        self._lock = asyncio.Lock()

    async def _run_async_source(
        self,
        source: AsyncBaseSource,
        limit: int,
    ) -> None:
        """Run a single async source and queue its documents.

        Args:
            source: The async source to run
            limit: Maximum documents to fetch
        """
        source_name = source.name
        self.progress.mark_started(source_name)

        try:
            await source.initialize()

            async for doc in source.fetch_documents(limit):
                await self._document_queue.put(doc)
                self.progress.increment(source_name)

                if self.progress_callback:
                    self.progress_callback(self.progress)

            self.progress.mark_completed(source_name)

        except Exception as e:
            logger.error(f"Source {source_name} failed: {e}")
            self.progress.mark_failed(source_name, str(e))

        finally:
            await source.cleanup()
            async with self._lock:
                self._active_sources -= 1
                if self._active_sources == 0:
                    # Signal completion
                    await self._document_queue.put(None)

    async def _run_sync_source_async(
        self,
        source: BaseSource,
        limit: int,
    ) -> None:
        """Run a sync source in a thread pool and queue its documents.

        Args:
            source: The sync source to run
            limit: Maximum documents to fetch
        """
        source_name = source.name
        self.progress.mark_started(source_name)

        try:
            source.initialize()

            # Run sync iterator in thread pool
            loop = asyncio.get_event_loop()

            def get_docs():
                return list(source.fetch_documents(limit))

            docs = await loop.run_in_executor(None, get_docs)

            for doc in docs:
                await self._document_queue.put(doc)
                self.progress.increment(source_name)

                if self.progress_callback:
                    self.progress_callback(self.progress)

            self.progress.mark_completed(source_name)

        except Exception as e:
            logger.error(f"Source {source_name} failed: {e}")
            self.progress.mark_failed(source_name, str(e))

        finally:
            source.cleanup()
            async with self._lock:
                self._active_sources -= 1
                if self._active_sources == 0:
                    # Signal completion
                    await self._document_queue.put(None)

    async def gather_all(
        self,
        sources: list[tuple[Union[AsyncBaseSource, BaseSource], int]],
    ) -> AsyncIterator[Document]:
        """Gather documents from all sources in parallel.

        Args:
            sources: List of (source, limit) tuples

        Yields:
            Documents as they arrive from any source
        """
        if not sources:
            return

        self.progress = OrchestratorProgress()
        self.progress.started_at = datetime.now(timezone.utc)
        self._document_queue = asyncio.Queue()

        # Initialize progress tracking
        for source, limit in sources:
            self.progress.add_source(source.name, limit)

        # Use semaphore if limiting concurrent sources
        semaphore = None
        if self.max_concurrent_sources:
            semaphore = asyncio.Semaphore(self.max_concurrent_sources)

        async def run_with_semaphore(source, limit):
            if semaphore:
                async with semaphore:
                    if isinstance(source, AsyncBaseSource):
                        await self._run_async_source(source, limit)
                    else:
                        await self._run_sync_source_async(source, limit)
            else:
                if isinstance(source, AsyncBaseSource):
                    await self._run_async_source(source, limit)
                else:
                    await self._run_sync_source_async(source, limit)

        # Start all sources
        self._active_sources = len(sources)
        tasks = []
        for source, limit in sources:
            task = asyncio.create_task(run_with_semaphore(source, limit))
            tasks.append(task)

        # Yield documents as they arrive
        while True:
            doc = await self._document_queue.get()
            if doc is None:
                break
            yield doc

        # Wait for all tasks to complete (they should be done)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def gather_sequential(
        self,
        sources: list[tuple[Union[AsyncBaseSource, BaseSource], int]],
    ) -> AsyncIterator[Document]:
        """Gather documents from sources sequentially (one at a time).

        Useful for debugging or when parallel execution causes issues.

        Args:
            sources: List of (source, limit) tuples

        Yields:
            Documents in order by source
        """
        self.progress = OrchestratorProgress()
        self.progress.started_at = datetime.now(timezone.utc)

        for source, limit in sources:
            self.progress.add_source(source.name, limit)

        for source, limit in sources:
            source_name = source.name
            self.progress.mark_started(source_name)

            try:
                if isinstance(source, AsyncBaseSource):
                    await source.initialize()
                    async for doc in source.fetch_documents(limit):
                        self.progress.increment(source_name)
                        yield doc
                    await source.cleanup()
                else:
                    source.initialize()
                    for doc in source.fetch_documents(limit):
                        self.progress.increment(source_name)
                        yield doc
                    source.cleanup()

                self.progress.mark_completed(source_name)

            except Exception as e:
                logger.error(f"Source {source_name} failed: {e}")
                self.progress.mark_failed(source_name, str(e))


class SourceRunner:
    """Utility class for running individual sources with proper lifecycle."""

    @staticmethod
    async def run_async(
        source: AsyncBaseSource,
        limit: int,
        on_document: Optional[Callable[[Document], Any]] = None,
    ) -> list[Document]:
        """Run an async source and collect documents.

        Args:
            source: The async source to run
            limit: Maximum documents to fetch
            on_document: Optional callback per document

        Returns:
            List of collected documents
        """
        documents = []

        try:
            await source.initialize()

            async for doc in source.fetch_documents(limit):
                documents.append(doc)
                if on_document:
                    result = on_document(doc)
                    if asyncio.iscoroutine(result):
                        await result

        finally:
            await source.cleanup()

        return documents

    @staticmethod
    def run_sync(
        source: BaseSource,
        limit: int,
        on_document: Optional[Callable[[Document], Any]] = None,
    ) -> list[Document]:
        """Run a sync source and collect documents.

        Args:
            source: The sync source to run
            limit: Maximum documents to fetch
            on_document: Optional callback per document

        Returns:
            List of collected documents
        """
        documents = []

        try:
            source.initialize()

            for doc in source.fetch_documents(limit):
                documents.append(doc)
                if on_document:
                    on_document(doc)

        finally:
            source.cleanup()

        return documents

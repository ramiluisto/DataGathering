"""Async Project Gutenberg source for gathering public domain books."""

import logging
import random
from typing import AsyncIterator, Optional

import httpx

from datagather.config import GutenbergConfig
from datagather.sources.base import AsyncBaseSource, Document
from datagather.utils.async_rate_limit import AsyncRateLimiter
from datagather.utils.async_retry import async_with_retry, check_response_for_retry
from datagather.utils.text import clean_text

logger = logging.getLogger(__name__)


class AsyncGutenbergSource(AsyncBaseSource):
    """Async Project Gutenberg source for gathering public domain books.

    Uses the Gutendex API (https://gutendex.com) with httpx async client.
    """

    name = "gutenberg"

    def __init__(self, config: GutenbergConfig, checkpoint_state: Optional[dict] = None):
        """Initialize Gutenberg source.

        Args:
            config: Gutenberg configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: GutenbergConfig = config
        self._rate_limiter = AsyncRateLimiter(requests_per_minute=10)
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize async HTTP client."""
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": "DataGather/0.1 (https://github.com/ramiluisto/DataGathering)",
            },
            timeout=60.0,  # Books can be large
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()

    @async_with_retry(max_attempts=3, delay=2.0)
    async def _search_books(self, page: int = 1, topic: Optional[str] = None) -> dict:
        """Search for books in Gutenberg.

        Args:
            page: Page number (1-indexed)
            topic: Optional topic/genre filter

        Returns:
            API response dictionary
        """
        await self._rate_limiter.acquire()

        params = {
            "languages": ",".join(self.config.languages),
            "page": page,
        }
        if topic:
            params["topic"] = topic

        response = await self._client.get(
            f"{self.config.api_url}/books",
            params=params,
        )
        check_response_for_retry(response)
        return response.json()

    @async_with_retry(max_attempts=3, delay=2.0)
    async def _get_book_text(self, book: dict) -> Optional[str]:
        """Download book text.

        Args:
            book: Book metadata from API

        Returns:
            Book text, or None if not available
        """
        formats = book.get("formats", {})

        # Prefer plain text formats
        text_urls = [
            formats.get("text/plain; charset=utf-8"),
            formats.get("text/plain; charset=us-ascii"),
            formats.get("text/plain"),
        ]

        for url in text_urls:
            if url and not url.endswith(".zip"):
                try:
                    await self._rate_limiter.acquire()
                    response = await self._client.get(url, timeout=60.0)
                    check_response_for_retry(response)
                    return response.text
                except Exception as e:
                    logger.debug(f"Failed to fetch {url}: {e}")
                    continue

        return None

    def _extract_metadata(self, book: dict) -> dict:
        """Extract metadata from book API response.

        Args:
            book: Book metadata from API

        Returns:
            Cleaned metadata dictionary
        """
        authors = book.get("authors", [])
        author_info = authors[0] if authors else {}

        return {
            "title": book.get("title", "Unknown"),
            "author": author_info.get("name", "Unknown"),
            "author_birth_year": author_info.get("birth_year"),
            "author_death_year": author_info.get("death_year"),
            "subjects": book.get("subjects", [])[:5],
            "bookshelves": book.get("bookshelves", [])[:5],
            "download_count": book.get("download_count", 0),
        }

    async def fetch_documents(self, limit: int) -> AsyncIterator[Document]:
        """Fetch books from Project Gutenberg asynchronously.

        Args:
            limit: Maximum number of books to fetch

        Yields:
            Document objects
        """
        if not self._client:
            await self.initialize()

        fetched = 0
        page = 1
        max_pages = 100  # Safety limit

        # Use genres if specified, otherwise search all
        genres = self.config.genres or [None]

        while fetched < limit and page <= max_pages:
            # Cycle through genres
            genre = genres[(page - 1) % len(genres)] if genres[0] else None

            try:
                data = await self._search_books(page=page, topic=genre)
            except Exception as e:
                logger.warning(f"Error searching books: {e}")
                page += 1
                continue

            books = data.get("results", [])
            if not books:
                break

            # Shuffle to get variety
            random.shuffle(books)

            for book in books:
                if fetched >= limit:
                    break

                book_id = str(book.get("id", ""))
                doc_id = f"gutenberg_{book_id}"

                # Skip if already processed
                if self._is_processed(doc_id):
                    continue

                try:
                    # Get book text
                    text = await self._get_book_text(book)
                    if not text:
                        logger.debug(f"No text available for book {book_id}")
                        continue

                    # Check minimum length
                    if len(text) < self.config.min_text_length:
                        logger.debug(f"Book {book_id} too short ({len(text)} chars)")
                        continue

                    # Clean the text
                    text = clean_text(text)

                    # Extract metadata
                    metadata = self._extract_metadata(book)

                    doc = Document(
                        id=doc_id,
                        source="gutenberg",
                        source_id=book_id,
                        text=text,
                        metadata={
                            "title": metadata["title"],
                            "author": metadata["author"],
                            "url": f"https://www.gutenberg.org/ebooks/{book_id}",
                        },
                        source_specific=metadata,
                    )

                    self._mark_processed(doc_id)
                    fetched += 1
                    yield doc

                except Exception as e:
                    logger.warning(f"Error processing book {book_id}: {e}")
                    continue

            page += 1

        logger.info(f"Gutenberg: fetched {fetched} books")

    def validate_config(self) -> list[str]:
        """Validate Gutenberg configuration."""
        errors = []
        if self.config.min_text_length < 0:
            errors.append("min_text_length must be non-negative")
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        return errors

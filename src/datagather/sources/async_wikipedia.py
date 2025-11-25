"""Async Wikipedia source for gathering random articles."""

import logging
from typing import AsyncIterator, Optional

import httpx

from datagather.config import WikipediaConfig
from datagather.sources.base import AsyncBaseSource, Document
from datagather.utils.async_rate_limit import AsyncRateLimiter
from datagather.utils.async_retry import async_with_retry, check_response_for_retry
from datagather.utils.text import clean_text

logger = logging.getLogger(__name__)


class AsyncWikipediaSource(AsyncBaseSource):
    """Async Wikipedia source for gathering random articles.

    Uses the MediaWiki API with httpx async client for non-blocking I/O.
    Supports filtering by categories and minimum article length.
    """

    name = "wikipedia"

    def __init__(self, config: WikipediaConfig, checkpoint_state: Optional[dict] = None):
        """Initialize Wikipedia source.

        Args:
            config: Wikipedia configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: WikipediaConfig = config
        self._rate_limiter = AsyncRateLimiter(requests_per_minute=30)
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize async HTTP client."""
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": "DataGather/0.1 (https://github.com/ramiluisto/DataGathering)",
            },
            timeout=30.0,
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()

    @async_with_retry(max_attempts=3, delay=1.0)
    async def _get_random_titles(self, count: int) -> list[str]:
        """Get random article titles from Wikipedia.

        Args:
            count: Number of titles to fetch

        Returns:
            List of article titles
        """
        await self._rate_limiter.acquire()
        response = await self._client.get(
            f"https://{self.config.language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "random",
                "rnnamespace": 0,  # Main namespace only
                "rnlimit": min(count, 20),  # API limit is 20
                "format": "json",
            },
        )
        check_response_for_retry(response)
        data = response.json()
        return [page["title"] for page in data.get("query", {}).get("random", [])]

    @async_with_retry(max_attempts=3, delay=1.0)
    async def _get_article_content(self, title: str) -> Optional[dict]:
        """Get article content from Wikipedia.

        Args:
            title: Article title

        Returns:
            Dictionary with article data, or None if not found
        """
        await self._rate_limiter.acquire()
        response = await self._client.get(
            f"https://{self.config.language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "extracts|info|categories",
                "explaintext": True,
                "inprop": "url",
                "cllimit": 10,
                "format": "json",
            },
        )
        check_response_for_retry(response)
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        page = list(pages.values())[0]
        if "missing" in page:
            return None

        text = page.get("extract", "")
        if len(text) < self.config.min_article_length:
            logger.debug(f"Skipping {title}: too short ({len(text)} chars)")
            return None

        # Check excluded categories
        if self.config.exclude_categories:
            categories = [c.get("title", "") for c in page.get("categories", [])]
            for exclude_cat in self.config.exclude_categories:
                if any(exclude_cat.lower() in cat.lower() for cat in categories):
                    logger.debug(f"Skipping {title}: excluded category")
                    return None

        return {
            "title": page.get("title", title),
            "text": text,
            "summary": text[:500] if text else "",
            "url": page.get("fullurl", f"https://{self.config.language}.wikipedia.org/wiki/{title}"),
            "page_id": str(page.get("pageid", "")),
            "categories": [c.get("title", "") for c in page.get("categories", [])],
        }

    async def fetch_documents(self, limit: int) -> AsyncIterator[Document]:
        """Fetch random Wikipedia articles asynchronously.

        Args:
            limit: Maximum number of articles to fetch

        Yields:
            Document objects
        """
        if not self._client:
            await self.initialize()

        fetched = 0
        attempts = 0
        max_attempts = limit * 3  # Allow for some failures

        while fetched < limit and attempts < max_attempts:
            # Get batch of random titles
            batch_size = min(20, limit - fetched)
            try:
                titles = await self._get_random_titles(batch_size)
            except Exception as e:
                logger.warning(f"Error fetching random titles: {e}")
                attempts += batch_size
                continue

            for title in titles:
                if fetched >= limit:
                    break

                attempts += 1
                doc_id = f"wikipedia_{title.replace(' ', '_')}"

                # Skip if already processed
                if self._is_processed(doc_id):
                    continue

                try:
                    article = await self._get_article_content(title)
                    if article is None:
                        continue

                    # Clean the text
                    text = clean_text(article["text"])

                    doc = Document(
                        id=doc_id,
                        source="wikipedia",
                        source_id=article["page_id"],
                        text=text,
                        metadata={
                            "title": article["title"],
                            "url": article["url"],
                            "language": self.config.language,
                        },
                        source_specific={
                            "page_id": article["page_id"],
                            "categories": article["categories"],
                            "summary": article["summary"][:500] if article["summary"] else "",
                        },
                    )

                    self._mark_processed(doc_id)
                    fetched += 1
                    yield doc

                except Exception as e:
                    logger.warning(f"Error fetching {title}: {e}")
                    continue

        logger.info(f"Wikipedia: fetched {fetched} articles in {attempts} attempts")

    def validate_config(self) -> list[str]:
        """Validate Wikipedia configuration."""
        errors = []
        if self.config.min_article_length < 0:
            errors.append("min_article_length must be non-negative")
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        return errors

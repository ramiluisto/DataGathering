"""Wikipedia source for gathering random articles."""

import logging
from typing import Iterator, Optional

import requests

from datagather.config import WikipediaConfig
from datagather.sources.base import BaseSource, Document
from datagather.utils.rate_limit import RateLimiter
from datagather.utils.text import clean_text

logger = logging.getLogger(__name__)

# Try to import wikipedia-api, fall back to requests if not available
try:
    import wikipediaapi

    HAS_WIKIPEDIA_API = True
except ImportError:
    HAS_WIKIPEDIA_API = False
    logger.warning("wikipedia-api not installed, using direct API calls")


class WikipediaSource(BaseSource):
    """Wikipedia source for gathering random articles.

    Uses the MediaWiki API to fetch random articles from Wikipedia.
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
        self._rate_limiter = RateLimiter(requests_per_minute=30)
        self._wiki = None
        self._session = None

    def initialize(self) -> None:
        """Initialize Wikipedia API client."""
        if HAS_WIKIPEDIA_API:
            self._wiki = wikipediaapi.Wikipedia(
                user_agent="DataGather/0.1 (https://github.com/ramiluisto/DataGathering)",
                language=self.config.language,
            )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "DataGather/0.1 (https://github.com/ramiluisto/DataGathering)",
            }
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._session:
            self._session.close()

    def _get_random_titles(self, count: int) -> list[str]:
        """Get random article titles from Wikipedia.

        Args:
            count: Number of titles to fetch

        Returns:
            List of article titles
        """
        self._rate_limiter.acquire()
        response = self._session.get(
            f"https://{self.config.language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "random",
                "rnnamespace": 0,  # Main namespace only
                "rnlimit": min(count, 20),  # API limit is 20
                "format": "json",
            },
        )
        response.raise_for_status()
        data = response.json()
        return [page["title"] for page in data.get("query", {}).get("random", [])]

    def _get_article_content(self, title: str) -> Optional[dict]:
        """Get article content from Wikipedia.

        Args:
            title: Article title

        Returns:
            Dictionary with article data, or None if not found
        """
        self._rate_limiter.acquire()

        if HAS_WIKIPEDIA_API and self._wiki:
            page = self._wiki.page(title)
            if not page.exists():
                return None

            # Check categories
            if self.config.exclude_categories:
                page_categories = set(page.categories.keys())
                for exclude_cat in self.config.exclude_categories:
                    if any(exclude_cat.lower() in cat.lower() for cat in page_categories):
                        logger.debug(f"Skipping {title}: excluded category")
                        return None

            text = page.text
            if len(text) < self.config.min_article_length:
                logger.debug(f"Skipping {title}: too short ({len(text)} chars)")
                return None

            return {
                "title": page.title,
                "text": text,
                "summary": page.summary,
                "url": page.fullurl,
                "page_id": str(page.pageid),
                "categories": list(page.categories.keys())[:10],  # Limit categories
            }
        else:
            # Fallback to direct API
            response = self._session.get(
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
            response.raise_for_status()
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

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch random Wikipedia articles.

        Args:
            limit: Maximum number of articles to fetch

        Yields:
            Document objects
        """
        if not self._session:
            self.initialize()

        fetched = 0
        attempts = 0
        max_attempts = limit * 3  # Allow for some failures

        while fetched < limit and attempts < max_attempts:
            # Get batch of random titles
            batch_size = min(20, limit - fetched)
            titles = self._get_random_titles(batch_size)

            for title in titles:
                if fetched >= limit:
                    break

                attempts += 1
                doc_id = f"wikipedia_{title.replace(' ', '_')}"

                # Skip if already processed
                if self._is_processed(doc_id):
                    continue

                try:
                    article = self._get_article_content(title)
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

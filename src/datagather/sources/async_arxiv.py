"""Async arXiv source for gathering academic papers."""

import logging
import xml.etree.ElementTree as ET
from typing import AsyncIterator, Optional

import httpx

from datagather.config import ArxivConfig
from datagather.sources.base import AsyncBaseSource, Document
from datagather.utils.async_rate_limit import AsyncDelayedRateLimiter
from datagather.utils.async_retry import async_with_retry, check_response_for_retry
from datagather.utils.text import clean_text

logger = logging.getLogger(__name__)

# arXiv Atom feed namespaces
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class AsyncArxivSource(AsyncBaseSource):
    """Async arXiv source for gathering academic papers.

    Uses the arXiv API directly with httpx for async HTTP.
    Respects arXiv's rate limit of 1 request per 3 seconds.
    """

    name = "arxiv"

    def __init__(self, config: ArxivConfig, checkpoint_state: Optional[dict] = None):
        """Initialize arXiv source.

        Args:
            config: arXiv configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: ArxivConfig = config
        # arXiv requires 3 second delay between requests
        self._rate_limiter = AsyncDelayedRateLimiter(delay_seconds=3.0)
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize async HTTP client."""
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": "DataGather/0.1 (https://github.com/ramiluisto/DataGathering)",
            },
            timeout=60.0,
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()

    def _build_query(self) -> str:
        """Build arXiv search query from configuration.

        Returns:
            Query string
        """
        if not self.config.categories:
            return "all"

        # Build category query
        cat_queries = [f"cat:{cat}" for cat in self.config.categories]
        return " OR ".join(cat_queries)

    def _parse_entry(self, entry: ET.Element) -> Optional[dict]:
        """Parse an Atom entry into a paper dict.

        Args:
            entry: XML Element for the entry

        Returns:
            Dictionary with paper data, or None if invalid
        """
        # Get entry ID (arxiv URL)
        id_elem = entry.find(f"{ATOM_NS}id")
        if id_elem is None or id_elem.text is None:
            return None

        entry_id = id_elem.text
        arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id.split("/")[-1]

        # Get title
        title_elem = entry.find(f"{ATOM_NS}title")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

        # Get summary/abstract
        summary_elem = entry.find(f"{ATOM_NS}summary")
        summary = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""

        # Get authors
        authors = []
        for author_elem in entry.findall(f"{ATOM_NS}author"):
            name_elem = author_elem.find(f"{ATOM_NS}name")
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)

        # Get categories
        categories = []
        primary_category = None
        for cat_elem in entry.findall(f"{ARXIV_NS}primary_category"):
            primary_category = cat_elem.get("term", "")
        for cat_elem in entry.findall(f"{ATOM_NS}category"):
            term = cat_elem.get("term", "")
            if term:
                categories.append(term)

        # Get dates
        published_elem = entry.find(f"{ATOM_NS}published")
        published = published_elem.text if published_elem is not None else None

        updated_elem = entry.find(f"{ATOM_NS}updated")
        updated = updated_elem.text if updated_elem is not None else None

        # Get links
        pdf_url = None
        for link_elem in entry.findall(f"{ATOM_NS}link"):
            if link_elem.get("title") == "pdf":
                pdf_url = link_elem.get("href")

        # Get optional fields
        doi_elem = entry.find(f"{ARXIV_NS}doi")
        doi = doi_elem.text if doi_elem is not None else None

        comment_elem = entry.find(f"{ARXIV_NS}comment")
        comment = comment_elem.text if comment_elem is not None else None

        journal_elem = entry.find(f"{ARXIV_NS}journal_ref")
        journal_ref = journal_elem.text if journal_elem is not None else None

        return {
            "arxiv_id": arxiv_id,
            "entry_id": entry_id,
            "title": title,
            "summary": summary,
            "authors": authors[:10],  # Limit authors
            "categories": categories,
            "primary_category": primary_category or (categories[0] if categories else ""),
            "published": published,
            "updated": updated,
            "pdf_url": pdf_url,
            "doi": doi,
            "comment": comment,
            "journal_ref": journal_ref,
        }

    @async_with_retry(max_attempts=3, delay=5.0, max_delay=30.0)
    async def _fetch_papers(
        self,
        query: str,
        start: int = 0,
        max_results: int = 100,
    ) -> list[dict]:
        """Fetch papers from arXiv API.

        Args:
            query: Search query
            start: Starting index
            max_results: Maximum results to return

        Returns:
            List of paper dictionaries
        """
        await self._rate_limiter.acquire()

        # Map sort_by config to API parameter
        sort_map = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
        }
        sort_by = sort_map.get(self.config.sort_by, "relevance")

        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        response = await self._client.get(
            "http://export.arxiv.org/api/query",
            params=params,
        )
        check_response_for_retry(response)

        # Parse XML response
        root = ET.fromstring(response.text)
        papers = []

        for entry in root.findall(f"{ATOM_NS}entry"):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return papers

    async def fetch_documents(self, limit: int) -> AsyncIterator[Document]:
        """Fetch papers from arXiv asynchronously.

        Args:
            limit: Maximum number of papers to fetch

        Yields:
            Document objects
        """
        if not self._client:
            await self.initialize()

        query = self._build_query()
        logger.info(f"arXiv query: {query}")

        fetched = 0
        start = 0
        batch_size = 100  # arXiv API max is 1000, but 100 is reasonable

        while fetched < limit:
            try:
                papers = await self._fetch_papers(
                    query=query,
                    start=start,
                    max_results=min(batch_size, (limit - fetched) * 2),
                )
            except Exception as e:
                logger.error(f"Error fetching papers: {e}")
                break

            if not papers:
                logger.info("No more papers found")
                break

            for paper in papers:
                if fetched >= limit:
                    break

                arxiv_id = paper["arxiv_id"]
                doc_id = f"arxiv_{arxiv_id.replace('.', '_').replace('/', '_')}"

                # Skip if already processed
                if self._is_processed(doc_id):
                    continue

                try:
                    # Get text based on content_type
                    if self.config.content_type == "abstract":
                        text = paper["summary"]
                    elif self.config.content_type == "full_text":
                        # Full text would require PDF download and extraction
                        # Use abstract + title as proxy
                        text = f"{paper['title']}\n\n{paper['summary']}"
                    else:  # both
                        text = f"{paper['title']}\n\n{paper['summary']}"

                    text = clean_text(text)

                    if not text or len(text) < 50:
                        continue

                    doc = Document(
                        id=doc_id,
                        source="arxiv",
                        source_id=arxiv_id,
                        text=text,
                        metadata={
                            "title": paper["title"],
                            "url": paper["entry_id"],
                            "authors": paper["authors"],
                        },
                        source_specific={
                            "arxiv_id": arxiv_id,
                            "categories": paper["categories"],
                            "primary_category": paper["primary_category"],
                            "published": paper["published"],
                            "updated": paper["updated"],
                            "doi": paper["doi"],
                            "pdf_url": paper["pdf_url"],
                            "comment": paper["comment"],
                            "journal_ref": paper["journal_ref"],
                        },
                    )

                    self._mark_processed(doc_id)
                    fetched += 1
                    yield doc

                except Exception as e:
                    logger.warning(f"Error processing paper {arxiv_id}: {e}")
                    continue

            start += len(papers)

        logger.info(f"arXiv: fetched {fetched} papers")

    def validate_config(self) -> list[str]:
        """Validate arXiv configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        return errors

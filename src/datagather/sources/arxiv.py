"""arXiv source for gathering academic papers."""

import logging
from typing import Iterator, Optional

from datagather.config import ArxivConfig
from datagather.sources.base import BaseSource, Document
from datagather.utils.rate_limit import DelayedRateLimiter
from datagather.utils.text import clean_text

logger = logging.getLogger(__name__)

# Try to import arxiv package
try:
    import arxiv

    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False
    logger.warning("arxiv package not installed")


class ArxivSource(BaseSource):
    """arXiv source for gathering academic papers.

    Uses the arxiv Python package to search and retrieve papers.
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
        self._rate_limiter = DelayedRateLimiter(delay_seconds=3.0)
        self._client = None

    def initialize(self) -> None:
        """Initialize arXiv client."""
        if not HAS_ARXIV:
            raise ImportError("arxiv package required: uv pip install arxiv")

        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3,
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

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

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch papers from arXiv.

        Args:
            limit: Maximum number of papers to fetch

        Yields:
            Document objects
        """
        if not self._client:
            self.initialize()

        query = self._build_query()
        logger.info(f"arXiv query: {query}")

        # Determine sort criterion
        sort_by_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_by = sort_by_map.get(self.config.sort_by, arxiv.SortCriterion.Relevance)

        search = arxiv.Search(
            query=query,
            max_results=limit * 2,  # Fetch extra to account for filtering
            sort_by=sort_by,
        )

        fetched = 0
        for result in self._client.results(search):
            if fetched >= limit:
                break

            arxiv_id = result.entry_id.split("/")[-1]
            doc_id = f"arxiv_{arxiv_id.replace('.', '_')}"

            # Skip if already processed
            if self._is_processed(doc_id):
                continue

            try:
                # Get text based on content_type
                if self.config.content_type == "abstract":
                    text = result.summary
                elif self.config.content_type == "full_text":
                    # Full text would require PDF download and extraction
                    # For now, use abstract + title as a reasonable proxy
                    text = f"{result.title}\n\n{result.summary}"
                else:  # both
                    text = f"{result.title}\n\n{result.summary}"

                text = clean_text(text)

                # Extract authors
                authors = [author.name for author in result.authors[:10]]

                doc = Document(
                    id=doc_id,
                    source="arxiv",
                    source_id=arxiv_id,
                    text=text,
                    metadata={
                        "title": result.title,
                        "url": result.entry_id,
                        "authors": authors,
                    },
                    source_specific={
                        "arxiv_id": arxiv_id,
                        "categories": result.categories,
                        "primary_category": result.primary_category,
                        "published": result.published.isoformat() if result.published else None,
                        "updated": result.updated.isoformat() if result.updated else None,
                        "doi": result.doi,
                        "pdf_url": result.pdf_url,
                        "comment": result.comment,
                        "journal_ref": result.journal_ref,
                    },
                )

                self._mark_processed(doc_id)
                fetched += 1
                yield doc

            except Exception as e:
                logger.warning(f"Error processing paper {arxiv_id}: {e}")
                continue

        logger.info(f"arXiv: fetched {fetched} papers")

    def validate_config(self) -> list[str]:
        """Validate arXiv configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_ARXIV:
            errors.append("arxiv package not installed")
        return errors

"""Legal source for gathering legal documents from HuggingFace datasets."""

import logging
from typing import Iterator, Optional

from datagather.config import LegalConfig
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


class LegalSource(BaseSource):
    """Legal documents source using Pile of Law from HuggingFace.

    The Pile of Law dataset includes:
    - courtlistener_opinions: Court opinions
    - federal_register: Federal Register documents
    - edgar: SEC EDGAR filings
    - And many more legal document types

    See: https://huggingface.co/datasets/pile-of-law/pile-of-law
    """

    name = "legal"

    def __init__(self, config: LegalConfig, checkpoint_state: Optional[dict] = None):
        """Initialize legal source.

        Args:
            config: Legal configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config: LegalConfig = config

    def initialize(self) -> None:
        """Initialize resources."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def _load_pile_of_law(self, subset: str, limit: int) -> Iterator[dict]:
        """Load Pile of Law subset.

        Args:
            subset: Dataset subset name
            limit: Maximum samples

        Yields:
            Legal document dictionaries
        """
        try:
            logger.info(f"Loading Pile of Law subset: {subset}")
            ds = load_dataset(
                "pile-of-law/pile-of-law",
                subset,
                streaming=True,
                split="train",
                trust_remote_code=True,
            )

            count = 0
            for sample in ds:
                if count >= limit:
                    return

                text = sample.get("text", "")
                if not text:
                    continue

                yield {
                    "text": text,
                    "subset": subset,
                    "meta": sample.get("meta", {}),
                    "created_timestamp": sample.get("created_timestamp", ""),
                    "downloaded_timestamp": sample.get("downloaded_timestamp", ""),
                }
                count += 1

        except Exception as e:
            logger.warning(f"Error loading pile-of-law/{subset}: {e}")

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch legal documents.

        Args:
            limit: Maximum number of documents to fetch

        Yields:
            Document objects
        """
        if not HAS_DATASETS:
            raise ImportError("datasets package required")

        fetched = 0

        # Get configured subsets
        pile_config = self.config.datasets.pile_of_law
        if pile_config.enabled:
            subsets = pile_config.subsets or ["courtlistener_opinions"]
            samples_per_subset = pile_config.samples // len(subsets)

            for subset in subsets:
                if fetched >= limit:
                    break

                for doc_data in self._load_pile_of_law(subset, samples_per_subset):
                    if fetched >= limit:
                        break

                    # Generate document ID
                    text = doc_data.get("text", "")
                    meta = doc_data.get("meta", {})
                    doc_id = f"legal_{subset}_{hash(text[:100])}"

                    # Skip if already processed
                    if self._is_processed(doc_id):
                        continue

                    # Clean text
                    text = clean_text(text)

                    # Extract metadata based on subset type
                    source_specific = {
                        "subset": subset,
                        "document_type": self._get_doc_type(subset),
                    }

                    # Add meta fields if available
                    if isinstance(meta, dict):
                        source_specific.update({
                            "court": meta.get("court", ""),
                            "date": meta.get("date", ""),
                            "case_name": meta.get("case_name", ""),
                            "jurisdiction": meta.get("jurisdiction", ""),
                        })

                    doc = Document(
                        id=doc_id,
                        source="legal",
                        source_id=str(hash(text[:100])),
                        text=text,
                        metadata={
                            "title": meta.get("case_name", "") if isinstance(meta, dict) else "",
                            "document_type": source_specific["document_type"],
                            "subset": subset,
                        },
                        source_specific=source_specific,
                    )

                    self._mark_processed(doc_id)
                    fetched += 1
                    yield doc

        logger.info(f"Legal: fetched {fetched} documents")

    def _get_doc_type(self, subset: str) -> str:
        """Map subset name to document type.

        Args:
            subset: Pile of Law subset name

        Returns:
            Human-readable document type
        """
        doc_types = {
            "courtlistener_opinions": "court_opinion",
            "federal_register": "federal_register",
            "edgar": "sec_filing",
            "cfr": "code_of_federal_regulations",
            "uscode": "us_code",
            "state_codes": "state_code",
            "scotus_oral_arguments": "oral_argument",
            "congressional_hearings": "hearing",
        }
        return doc_types.get(subset, "legal_document")

    def validate_config(self) -> list[str]:
        """Validate legal configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

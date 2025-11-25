"""Reviews source for gathering product reviews (Amazon, etc.)."""

import logging
from typing import Iterator, Optional

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


class ReviewsConfig:
    """Reviews source configuration."""

    def __init__(
        self,
        enabled: bool = True,
        samples: int = 100,
        dataset: str = "McAuley-Lab/Amazon-Reviews-2023",
        category: str = "All_Beauty",
        min_rating: int = 1,
        max_rating: int = 5,
        min_review_length: int = 100,
        include_title: bool = True,
        include_helpful_votes: bool = True,
    ):
        self.enabled = enabled
        self.samples = samples
        self.dataset = dataset
        self.category = category
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_review_length = min_review_length
        self.include_title = include_title
        self.include_helpful_votes = include_helpful_votes


class ReviewsSource(BaseSource):
    """Reviews source for gathering product reviews.

    Uses HuggingFace datasets for Amazon reviews and similar.
    Provides opinionated consumer text with sentiment.
    """

    name = "reviews"

    # Available categories for Amazon Reviews 2023
    AMAZON_CATEGORIES = [
        "All_Beauty",
        "Amazon_Fashion",
        "Appliances",
        "Arts_Crafts_and_Sewing",
        "Automotive",
        "Baby_Products",
        "Beauty_and_Personal_Care",
        "Books",
        "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories",
        "Clothing_Shoes_and_Jewelry",
        "Digital_Music",
        "Electronics",
        "Gift_Cards",
        "Grocery_and_Gourmet_Food",
        "Handmade_Products",
        "Health_and_Household",
        "Health_and_Personal_Care",
        "Home_and_Kitchen",
        "Industrial_and_Scientific",
        "Kindle_Store",
        "Magazine_Subscriptions",
        "Movies_and_TV",
        "Musical_Instruments",
        "Office_Products",
        "Patio_Lawn_and_Garden",
        "Pet_Supplies",
        "Software",
        "Sports_and_Outdoors",
        "Subscription_Boxes",
        "Tools_and_Home_Improvement",
        "Toys_and_Games",
        "Video_Games",
    ]

    def __init__(self, config, checkpoint_state: Optional[dict] = None):
        """Initialize Reviews source.

        Args:
            config: Reviews configuration
            checkpoint_state: Optional checkpoint state to resume from
        """
        super().__init__(config, checkpoint_state)
        self.config = config
        self._dataset = None
        self._iterator = None

    def initialize(self) -> None:
        """Initialize dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required: uv pip install datasets")

        dataset_name = self.config.dataset
        category = self.config.category

        try:
            logger.info(f"Loading reviews dataset: {dataset_name} ({category})")

            # Try the Amazon Reviews 2023 dataset first
            if "Amazon-Reviews-2023" in dataset_name:
                self._dataset = load_dataset(
                    dataset_name,
                    f"raw_review_{category}",
                    split="full",
                    streaming=True,
                    trust_remote_code=True,
                )
            else:
                # Generic fallback
                self._dataset = load_dataset(
                    dataset_name,
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )

            self._iterator = iter(self._dataset)

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            # Fall back to amazon_polarity dataset (simpler, always available)
            logger.info("Falling back to amazon_polarity dataset")
            self._dataset = load_dataset(
                "amazon_polarity",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self._iterator = iter(self._dataset)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._dataset = None
        self._iterator = None

    def _format_review(self, item: dict) -> Optional[tuple[str, dict]]:
        """Format a review item into text.

        Args:
            item: Dataset item

        Returns:
            Tuple of (formatted_text, metadata) or None if invalid
        """
        # Handle Amazon Reviews 2023 format
        if "text" in item and "rating" in item:
            text = item.get("text", "")
            title = item.get("title", "")
            rating = item.get("rating", 0)
            helpful_vote = item.get("helpful_vote", 0)
            verified = item.get("verified_purchase", False)
            asin = item.get("asin", "")
            parent_asin = item.get("parent_asin", "")

            # Filter by rating
            if rating < self.config.min_rating or rating > self.config.max_rating:
                return None

            # Build review text
            review_text = ""
            if self.config.include_title and title:
                review_text = f"{title}\n\n{text}"
            else:
                review_text = text

            metadata = {
                "rating": rating,
                "helpful_votes": helpful_vote,
                "verified_purchase": verified,
                "asin": asin,
                "parent_asin": parent_asin,
            }

        # Handle amazon_polarity format
        elif "content" in item and "label" in item:
            text = item.get("content", "")
            title = item.get("title", "")
            label = item.get("label", 0)  # 0 = negative, 1 = positive

            # Convert label to rating-like value
            rating = 5 if label == 1 else 1

            if self.config.include_title and title:
                review_text = f"{title}\n\n{text}"
            else:
                review_text = text

            metadata = {
                "rating": rating,
                "sentiment": "positive" if label == 1 else "negative",
            }

        else:
            return None

        # Clean the text
        review_text = clean_text(review_text)

        # Check minimum length
        if len(review_text) < self.config.min_review_length:
            return None

        return review_text, metadata

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch reviews from dataset.

        Args:
            limit: Maximum number of reviews to fetch

        Yields:
            Document objects
        """
        if not self._iterator:
            self.initialize()

        fetched = 0
        attempts = 0
        max_attempts = limit * 10  # Reviews can be short, need more filtering

        while fetched < limit and attempts < max_attempts:
            try:
                item = next(self._iterator)
                attempts += 1
            except StopIteration:
                logger.info("Reached end of reviews dataset")
                break

            # Create document ID
            if "user_id" in item and "timestamp" in item:
                review_id = f"{item['user_id']}_{item['timestamp']}"
            elif "asin" in item:
                review_id = f"{item.get('asin', '')}_{attempts}"
            else:
                review_id = str(attempts)

            doc_id = f"review_{review_id}"

            if self._is_processed(doc_id):
                continue

            result = self._format_review(item)
            if result is None:
                continue

            text, metadata = result

            doc = Document(
                id=doc_id,
                source="reviews",
                source_id=review_id,
                text=text,
                metadata={
                    "rating": metadata.get("rating", 0),
                    "category": self.config.category,
                },
                source_specific=metadata,
            )

            self._mark_processed(doc_id)
            fetched += 1
            yield doc

        logger.info(f"Reviews: fetched {fetched} reviews in {attempts} attempts")

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if self.config.min_rating > self.config.max_rating:
            errors.append("min_rating must be <= max_rating")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

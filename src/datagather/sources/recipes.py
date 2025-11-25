"""Recipes source for gathering cooking recipes and instructions."""

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


class RecipesConfig:
    """Recipes source configuration."""

    def __init__(
        self,
        enabled: bool = True,
        samples: int = 100,
        dataset: str = "recipe_nlg",
        min_ingredients: int = 3,
        min_instructions_length: int = 100,
        include_nutrition: bool = False,
    ):
        self.enabled = enabled
        self.samples = samples
        self.dataset = dataset
        self.min_ingredients = min_ingredients
        self.min_instructions_length = min_instructions_length
        self.include_nutrition = include_nutrition


class RecipesSource(BaseSource):
    """Recipes source for gathering cooking recipes.

    Uses HuggingFace datasets for RecipeNLG and similar.
    Provides procedural instructional text with domain-specific vocabulary.
    """

    name = "recipes"

    # Available recipe datasets
    DATASETS = {
        "recipe_nlg": "recipe_nlg",
        "all_recipes": "Shengtao/all-recipes",
        "food_com": "Hieu-Pham/food_com",
    }

    def __init__(self, config, checkpoint_state: Optional[dict] = None):
        """Initialize Recipes source.

        Args:
            config: Recipes configuration
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

        try:
            logger.info(f"Loading recipes dataset: {dataset_name}")
            self._dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self._iterator = iter(self._dataset)

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            # Try fallback datasets
            for fallback in ["mbien/recipe_nlg", "Hieu-Pham/food_com"]:
                try:
                    logger.info(f"Trying fallback: {fallback}")
                    self._dataset = load_dataset(
                        fallback,
                        split="train",
                        streaming=True,
                        trust_remote_code=True,
                    )
                    self._iterator = iter(self._dataset)
                    break
                except Exception:
                    continue

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._dataset = None
        self._iterator = None

    def _format_recipe(self, item: dict) -> Optional[tuple[str, dict]]:
        """Format a recipe item into text.

        Args:
            item: Dataset item

        Returns:
            Tuple of (formatted_text, metadata) or None if invalid
        """
        # Handle RecipeNLG format
        if "title" in item and "ingredients" in item and "directions" in item:
            title = item.get("title", "")
            ingredients = item.get("ingredients", [])
            directions = item.get("directions", [])
            ner = item.get("NER", [])  # Named entities (parsed ingredients)

            # Handle string vs list
            if isinstance(ingredients, str):
                ingredients = [ing.strip() for ing in ingredients.split("\n") if ing.strip()]
            if isinstance(directions, str):
                directions = [d.strip() for d in directions.split("\n") if d.strip()]

            # Check minimum requirements
            if len(ingredients) < self.config.min_ingredients:
                return None

            # Format the recipe
            parts = [f"# {title}", "", "## Ingredients"]
            for ing in ingredients:
                parts.append(f"- {ing}")

            parts.extend(["", "## Instructions"])
            for i, step in enumerate(directions, 1):
                parts.append(f"{i}. {step}")

            text = "\n".join(parts)

            metadata = {
                "title": title,
                "ingredient_count": len(ingredients),
                "step_count": len(directions),
                "ingredients": ingredients[:10],  # Limit for storage
                "named_entities": ner[:20] if ner else [],
            }

        # Handle food.com format
        elif "name" in item and "steps" in item:
            name = item.get("name", "")
            steps = item.get("steps", [])
            ingredients_list = item.get("ingredients", [])
            description = item.get("description", "")

            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("\n") if s.strip()]
            if isinstance(ingredients_list, str):
                ingredients_list = [i.strip() for i in ingredients_list.split("\n") if i.strip()]

            if len(ingredients_list) < self.config.min_ingredients:
                return None

            parts = [f"# {name}"]
            if description:
                parts.extend(["", description])

            parts.extend(["", "## Ingredients"])
            for ing in ingredients_list:
                parts.append(f"- {ing}")

            parts.extend(["", "## Instructions"])
            for i, step in enumerate(steps, 1):
                parts.append(f"{i}. {step}")

            text = "\n".join(parts)

            metadata = {
                "title": name,
                "description": description,
                "ingredient_count": len(ingredients_list),
                "step_count": len(steps),
                "minutes": item.get("minutes"),
                "nutrition": item.get("nutrition"),
            }

        else:
            return None

        # Clean and validate
        text = clean_text(text)

        # Check minimum instruction length
        instructions_text = text.split("## Instructions")[-1] if "## Instructions" in text else text
        if len(instructions_text) < self.config.min_instructions_length:
            return None

        return text, metadata

    def fetch_documents(self, limit: int) -> Iterator[Document]:
        """Fetch recipes from dataset.

        Args:
            limit: Maximum number of recipes to fetch

        Yields:
            Document objects
        """
        if not self._iterator:
            self.initialize()

        fetched = 0
        attempts = 0
        max_attempts = limit * 5

        while fetched < limit and attempts < max_attempts:
            try:
                item = next(self._iterator)
                attempts += 1
            except StopIteration:
                logger.info("Reached end of recipes dataset")
                break

            # Create document ID
            if "id" in item:
                recipe_id = str(item["id"])
            elif "title" in item:
                recipe_id = item["title"].replace(" ", "_")[:50]
            elif "name" in item:
                recipe_id = item["name"].replace(" ", "_")[:50]
            else:
                recipe_id = str(attempts)

            doc_id = f"recipe_{recipe_id}"

            if self._is_processed(doc_id):
                continue

            result = self._format_recipe(item)
            if result is None:
                continue

            text, metadata = result

            doc = Document(
                id=doc_id,
                source="recipes",
                source_id=recipe_id,
                text=text,
                metadata={
                    "title": metadata.get("title", ""),
                    "type": "recipe",
                },
                source_specific=metadata,
            )

            self._mark_processed(doc_id)
            fetched += 1
            yield doc

        logger.info(f"Recipes: fetched {fetched} recipes in {attempts} attempts")

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.config.samples <= 0:
            errors.append("samples must be positive")
        if self.config.min_ingredients < 0:
            errors.append("min_ingredients must be non-negative")
        if not HAS_DATASETS:
            errors.append("datasets package not installed")
        return errors

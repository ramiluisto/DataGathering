"""Configuration models using Pydantic v2."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    retry_exponential_base: float = 2.0


class DateRange(BaseModel):
    """Date range for filtering."""

    start: Optional[str] = None
    end: Optional[str] = None


class TimePeriod(BaseModel):
    """Time period with weight for sampling."""

    start: int
    end: int
    weight: float = 1.0


# =============================================================================
# Source-specific configurations
# =============================================================================


class WikipediaConfig(BaseModel):
    """Wikipedia source configuration."""

    enabled: bool = True
    samples: int = 100
    language: str = "en"
    min_article_length: int = 500
    categories: list[str] = Field(default_factory=list)
    exclude_categories: list[str] = Field(
        default_factory=lambda: ["Disambiguation pages", "Stub articles"]
    )


class GutenbergConfig(BaseModel):
    """Project Gutenberg source configuration."""

    enabled: bool = True
    samples: int = 50
    api_url: str = "https://gutendex.com"
    languages: list[str] = Field(default_factory=lambda: ["en"])
    genres: list[str] = Field(default_factory=lambda: ["Fiction", "Science", "History"])
    time_periods: list[TimePeriod] = Field(default_factory=list)
    min_text_length: int = 10000


class CodeConfig(BaseModel):
    """Code source configuration (The Stack / CodeParrot)."""

    enabled: bool = True
    samples: int = 200
    dataset: str = "codeparrot/github-code"
    streaming: bool = True
    languages: list[str] = Field(
        default_factory=lambda: ["Python", "JavaScript", "Java", "Go", "Rust", "C++"]
    )
    samples_per_language: Optional[int] = None
    min_file_size: int = 100
    max_file_size: int = 100000
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["**/test/**", "**/*_test.*", "**/vendor/**"]
    )


class ArxivConfig(BaseModel):
    """arXiv source configuration."""

    enabled: bool = True
    samples: int = 100
    categories: list[str] = Field(
        default_factory=lambda: ["cs.CL", "cs.AI", "cs.LG", "stat.ML", "physics", "math"]
    )
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    content_type: Literal["abstract", "full_text", "both"] = "abstract"
    sort_by: str = "relevance"
    rate_limit: Optional[RateLimitConfig] = None


class RedditConfig(BaseModel):
    """Reddit source configuration (Pushshift archives)."""

    enabled: bool = True
    samples: int = 200
    data_source: Literal["pushshift_archive", "huggingface"] = "huggingface"
    archive_path: Optional[str] = None
    subreddits: list[str] = Field(
        default_factory=lambda: [
            "AskScience",
            "ExplainLikeImFive",
            "WritingPrompts",
            "TodayILearned",
            "ChangeMyView",
        ]
    )
    min_score: int = 10
    min_comment_length: int = 100
    content_type: Literal["submissions", "comments", "both"] = "both"
    date_range: Optional[DateRange] = None


class TwitterConfig(BaseModel):
    """Twitter source configuration (GESIS TweetsKB archives)."""

    enabled: bool = False  # Disabled by default - requires manual setup
    samples: int = 100
    data_source: str = "tweetskb"
    min_tweet_length: int = 50
    exclude_retweets: bool = True


class CommonCrawlConfig(BaseModel):
    """Common Crawl configuration for news."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    publishers: list[str] = Field(default_factory=list)


class NewsConfig(BaseModel):
    """News source configuration (Common Crawl)."""

    enabled: bool = True
    samples: int = 100
    source_type: Literal["commoncrawl", "huggingface"] = "huggingface"
    commoncrawl: CommonCrawlConfig = Field(default_factory=CommonCrawlConfig)
    min_article_length: int = 500
    include_title: bool = True
    include_authors: bool = False


class PileOfLawConfig(BaseModel):
    """Pile of Law dataset configuration."""

    enabled: bool = True
    samples: int = 70
    subsets: list[str] = Field(
        default_factory=lambda: ["courtlistener_opinions", "federal_register"]
    )
    streaming: bool = True


class USPTOConfig(BaseModel):
    """USPTO patents configuration."""

    enabled: bool = True
    samples: int = 30
    content_type: Literal["abstract", "claims", "description"] = "abstract"


class LegalDatasetsConfig(BaseModel):
    """Legal datasets sub-configuration."""

    pile_of_law: PileOfLawConfig = Field(default_factory=PileOfLawConfig)
    uspto: USPTOConfig = Field(default_factory=USPTOConfig)


class LegalConfig(BaseModel):
    """Legal source configuration."""

    enabled: bool = True
    samples: int = 100
    datasets: LegalDatasetsConfig = Field(default_factory=LegalDatasetsConfig)


class OyezConfig(BaseModel):
    """Oyez Supreme Court transcripts configuration."""

    enabled: bool = True
    samples: int = 60
    year_range: list[int] = Field(default_factory=lambda: [1990, 2023])
    include_metadata: bool = True


class DialogConfig(BaseModel):
    """Dialog datasets configuration."""

    enabled: bool = True
    samples: int = 40
    sources: list[str] = Field(default_factory=lambda: ["daily_dialog", "topical_chat"])


class ConversationalDatasetsConfig(BaseModel):
    """Conversational datasets sub-configuration."""

    oyez: OyezConfig = Field(default_factory=OyezConfig)
    dialog: DialogConfig = Field(default_factory=DialogConfig)


class ConversationalConfig(BaseModel):
    """Conversational source configuration."""

    enabled: bool = True
    samples: int = 100
    datasets: ConversationalDatasetsConfig = Field(default_factory=ConversationalDatasetsConfig)


class StackOverflowConfig(BaseModel):
    """Stack Overflow Q&A source configuration."""

    enabled: bool = True
    samples: int = 100
    dataset: str = "koutch/stackoverflow_python"
    min_score: int = 5
    min_answer_score: int = 3
    include_code: bool = True
    languages: list[str] = Field(default_factory=lambda: ["python"])


class ReviewsConfig(BaseModel):
    """Product reviews source configuration."""

    enabled: bool = True
    samples: int = 100
    dataset: str = "McAuley-Lab/Amazon-Reviews-2023"
    category: str = "All_Beauty"
    min_rating: int = 1
    max_rating: int = 5
    min_review_length: int = 100
    include_title: bool = True


class SubtitlesConfig(BaseModel):
    """Movie/TV subtitles source configuration."""

    enabled: bool = True
    samples: int = 100
    dataset: str = "open_subtitles"
    language: str = "en"
    min_length: int = 500
    combine_utterances: bool = True
    max_utterances_per_doc: int = 50


class RecipesConfig(BaseModel):
    """Cooking recipes source configuration."""

    enabled: bool = True
    samples: int = 100
    dataset: str = "recipe_nlg"
    min_ingredients: int = 3
    min_instructions_length: int = 100


# =============================================================================
# Main configuration
# =============================================================================


class SourcesConfig(BaseModel):
    """All data sources configuration."""

    wikipedia: WikipediaConfig = Field(default_factory=WikipediaConfig)
    gutenberg: GutenbergConfig = Field(default_factory=GutenbergConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    twitter: TwitterConfig = Field(default_factory=TwitterConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    legal: LegalConfig = Field(default_factory=LegalConfig)
    conversational: ConversationalConfig = Field(default_factory=ConversationalConfig)
    stackoverflow: StackOverflowConfig = Field(default_factory=StackOverflowConfig)
    reviews: ReviewsConfig = Field(default_factory=ReviewsConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)
    recipes: RecipesConfig = Field(default_factory=RecipesConfig)


class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""

    enabled: bool = True
    save_interval: int = 50
    directory: str = "./data/checkpoints"


class AsyncConfig(BaseModel):
    """Async execution configuration."""

    enabled: bool = True
    max_concurrent_sources: Optional[int] = None  # None = all sources in parallel
    use_async_sources: bool = True  # Use async versions of sources when available


class GatherConfig(BaseModel):
    """Main configuration for data gathering."""

    output_dir: str = "./data"
    default_samples_per_source: int = 100
    save_full_documents: bool = True
    save_chunks: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: Literal["token", "sentence", "paragraph"] = "token"
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    async_config: AsyncConfig = Field(default_factory=AsyncConfig)

    @field_validator("output_dir", "checkpoint")
    @classmethod
    def ensure_paths(cls, v):
        """Ensure paths are valid."""
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "GatherConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))

    @classmethod
    def default(cls) -> "GatherConfig":
        """Return default configuration."""
        return cls()

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_enabled_sources(self) -> list[str]:
        """Return list of enabled source names."""
        enabled = []
        sources_dict = self.sources.model_dump()
        for name, config in sources_dict.items():
            if config.get("enabled", False):
                enabled.append(name)
        return enabled

    def get_source_samples(self, source_name: str, override: Optional[int] = None) -> int:
        """Get the number of samples for a source."""
        if override is not None:
            return override
        source_config = getattr(self.sources, source_name, None)
        if source_config and hasattr(source_config, "samples"):
            return source_config.samples
        return self.default_samples_per_source

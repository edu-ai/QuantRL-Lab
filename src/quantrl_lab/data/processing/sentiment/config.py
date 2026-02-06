"""Configuration for sentiment analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SentimentConfig:
    """
    Base configuration for sentiment analysis.

    This config is provider-agnostic and contains common settings.
    Provider-specific settings should be passed via provider_kwargs.
    """

    text_column: str = "headline"
    date_column: str = "created_at"
    sentiment_score_column: str = "sentiment_score"
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """
        Convert config to dictionary.

        Returns:
            Dict: Dictionary representation of config.
        """
        return {
            "text_column": self.text_column,
            "date_column": self.date_column,
            "sentiment_score_column": self.sentiment_score_column,
            "provider_kwargs": self.provider_kwargs,
        }


@dataclass
class HuggingFaceConfig:
    """HuggingFace-specific configuration for sentiment analysis."""

    model_name: str = "ProsusAI/finbert"
    device: int = -1  # -1 for CPU, 0 for GPU
    max_length: Optional[int] = None
    truncation: bool = True
    top_k: int = 1  # `return_all_scores` is deprecated
    batch_size: int = 1  # Batch size for inference

    # Supported models for validation
    SUPPORTED_MODELS: List[str] = field(
        default_factory=lambda: [
            "ProsusAI/finbert",
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            "ElKulako/cryptobert",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        ]
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device < -1:
            raise ValueError("device must be -1 (CPU) or >= 0 (GPU)")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def to_dict(self) -> Dict:
        """
        Convert config to dictionary.

        Returns:
            Dict: Dictionary representation of config.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
        }

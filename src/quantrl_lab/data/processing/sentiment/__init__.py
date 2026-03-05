"""
Sentiment analysis module.

This module provides providers and configuration for sentiment analysis.
"""

from .config import HuggingFaceConfig, SentimentConfig
from .provider import HuggingFaceProvider, SentimentProvider

__all__ = [
    "SentimentConfig",
    "HuggingFaceConfig",
    "SentimentProvider",
    "HuggingFaceProvider",
]

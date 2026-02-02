"""Sentiment enrichment processing step."""

import pandas as pd
from rich.console import Console

from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator
from quantrl_lab.data.processing.processor import ProcessingMetadata
from quantrl_lab.data.processing.sentiment import SentimentConfig, SentimentProvider

console = Console()


class SentimentEnrichmentStep:
    """
    Add news sentiment scores to DataFrame.

    This step enriches OHLCV data with sentiment scores computed from
    news data. Requires news_data to be provided.

    Example:
        >>> step = SentimentEnrichmentStep(
        ...     news_data=news_df,
        ...     provider=HuggingFaceProvider(),
        ...     fillna_strategy="neutral"
        ... )
        >>> result = step.process(df, metadata)
    """

    def __init__(
        self,
        news_data: pd.DataFrame,
        provider: SentimentProvider = None,
        config: SentimentConfig = None,
        fillna_strategy: str = "neutral",
    ):
        """
        Initialize sentiment enrichment step.

        Args:
            news_data: DataFrame with news articles
            provider: Sentiment analysis provider (default: HuggingFaceProvider)
            config: Sentiment configuration
            fillna_strategy: Strategy for filling missing scores ("neutral" or "fill_forward")
        """
        self.news_data = news_data
        self.provider = provider
        self.config = config or SentimentConfig()
        self.fillna_strategy = fillna_strategy

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Add sentiment scores to DataFrame.

        Args:
            data: Input OHLCV DataFrame
            metadata: Processing metadata (updated with sentiment flag)

        Returns:
            DataFrame with sentiment scores added

        Raises:
            ValueError: If news_data is empty or invalid
        """
        if self.news_data is None or self.news_data.empty:
            console.print("[yellow]⚠️  No news data provided. Skipping sentiment analysis.[/yellow]")
            return data

        try:
            generator = SentimentFeatureGenerator(
                self.provider,
                self.config,
                self.news_data,
                self.fillna_strategy,
            )
            result = generator.generate(data)

            # Update metadata
            metadata.news_sentiment_applied = True
            metadata.fillna_strategy = self.fillna_strategy

            return result
        except ValueError as e:
            raise e
        except Exception as e:
            console.print(f"[red]❌ Failed to add sentiment data: {e}[/red]")
            return data

    def get_step_name(self) -> str:
        """Return step name."""
        return "Sentiment Enrichment"

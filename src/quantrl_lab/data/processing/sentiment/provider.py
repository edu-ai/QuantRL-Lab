"""Sentiment analysis providers."""

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import pandas as pd
from rich.console import Console

from .config import HuggingFaceConfig, SentimentConfig

if TYPE_CHECKING:
    from transformers import Pipeline

console = Console()


@runtime_checkable
class SentimentProvider(Protocol):
    """Protocol for sentiment analysis providers."""

    def analyze(self, text_data: pd.DataFrame, config: SentimentConfig) -> pd.DataFrame:
        """
        Analyze sentiment of text data.

        Args:
            text_data (pd.DataFrame): DataFrame containing text to analyze.
                Must include columns specified in config (text_column, date_column).
            config (SentimentConfig): Configuration for sentiment analysis.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - Date: Date of the text (aggregated by day)
                - sentiment_score: Average sentiment score for that date
        """
        ...


class HuggingFaceProvider:
    """
    Sentiment analysis using HuggingFace transformers.

    This provider uses pre-trained transformer models for financial
    sentiment analysis. Requires 'transformers' and 'torch' packages.

    Example:
        >>> from quantrl_lab.data.processing.sentiment import HuggingFaceProvider, SentimentConfig, HuggingFaceConfig
        >>> provider = HuggingFaceProvider(HuggingFaceConfig(model_name="ProsusAI/finbert", batch_size=32))
        >>> config = SentimentConfig(text_column="headline", date_column="created_at")
        >>> sentiment_scores = provider.analyze(news_df, config)
    """

    def __init__(self, hf_config: Optional[HuggingFaceConfig] = None):
        """
        Initialize HuggingFace sentiment provider.

        Args:
            hf_config (Optional[HuggingFaceConfig]): HuggingFace-specific configuration.
                If None, uses default configuration.
        """
        self.hf_config = hf_config or HuggingFaceConfig()
        self._pipeline: Optional["Pipeline"] = None

    def _get_pipeline(self) -> "Pipeline":
        """
        Lazy initialization of sentiment analysis pipeline.

        Returns:
            Pipeline: HuggingFace sentiment analysis pipeline.

        Raises:
            ImportError: If transformers or torch not installed.
            RuntimeError: If model loading fails.
        """
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as e:
                raise ImportError(
                    "Sentiment analysis requires 'transformers' and 'torch'. " "Install them with: uv sync --extra ml"
                ) from e

            try:
                # Use GPU if available and device is set to 0
                device = 0 if torch.cuda.is_available() and self.hf_config.device == 0 else -1

                pipeline_kwargs = {
                    "model": self.hf_config.model_name,
                    "tokenizer": self.hf_config.model_name,
                    "device": device,
                    "truncation": self.hf_config.truncation,
                    "top_k": self.hf_config.top_k,
                }

                if self.hf_config.max_length:
                    pipeline_kwargs["max_length"] = self.hf_config.max_length

                self._pipeline = pipeline("sentiment-analysis", **pipeline_kwargs)
                console.print(
                    f"[green]✓ Sentiment analysis pipeline initialized with model: "
                    f"{self.hf_config.model_name}[/green]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load sentiment model: {e}")

        return self._pipeline

    def analyze(self, text_data: pd.DataFrame, config: SentimentConfig) -> pd.DataFrame:
        """
        Analyze sentiment of text data using HuggingFace models.

        If sentiment scores are already present in the data, they will be used.
        Otherwise, the HuggingFace model will calculate them.

        Args:
            text_data (pd.DataFrame): DataFrame containing text articles.
            config (SentimentConfig): Configuration for sentiment analysis.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - Date: Date of the text (aggregated by day)
                - sentiment_score: Average sentiment score for that date

        Raises:
            ValueError: If text_data is empty or required columns missing.
            ValueError: If no valid text data found.
            RuntimeError: If sentiment analysis fails.
        """
        # === Input validation ===
        if text_data.empty:
            raise ValueError("Text data cannot be empty")

        required_columns = [config.date_column]
        # If sentiment score is not present, text column is required for analysis
        if config.sentiment_score_column not in text_data.columns:
            required_columns.append(config.text_column)

        for col in required_columns:
            if col not in text_data.columns:
                raise ValueError(f"Required column '{col}' not found. Available columns: {list(text_data.columns)}")

        # Make a copy to avoid modifying original
        text_data = text_data.copy()

        # === Process sentiment scores ===
        if config.sentiment_score_column in text_data.columns:
            console.print("[green]✓ Using pre-existing sentiment scores.[/green]")
            # Ensure the sentiment score column is numeric
            text_data[config.sentiment_score_column] = pd.to_numeric(
                text_data[config.sentiment_score_column], errors="coerce"
            )
        else:
            console.print("[cyan]Calculating sentiment scores using HuggingFace model...[/cyan]")
            # === Initialize pipeline ===
            sentiment_pipeline = self._get_pipeline()

            # === Prepare texts ===
            texts_to_analyze = text_data[config.text_column].fillna("").astype(str).tolist()

            if not texts_to_analyze:
                raise ValueError("No valid text data found for sentiment analysis")

            # === Run sentiment analysis ===
            sentiments = sentiment_pipeline(
                texts_to_analyze,
                batch_size=self.hf_config.batch_size,
                truncation=self.hf_config.truncation,
            )

            # Handle cases where each result might itself be a list
            scores = []
            for result in sentiments:
                if isinstance(result, list):
                    # Ensure result is not empty
                    if result:
                        scores.append(result[0].get("score", 0.0))
                    else:
                        scores.append(0.0)
                else:
                    scores.append(result.get("score", 0.0))

            text_data[config.sentiment_score_column] = scores

        # === Process date column and aggregate ===
        text_data[config.date_column] = pd.to_datetime(text_data[config.date_column]).dt.date

        # Group by date and calculate mean sentiment score
        aggregated = text_data.groupby(config.date_column).agg({config.sentiment_score_column: "mean"}).reset_index()

        # Rename columns to standard format
        aggregated.rename(
            columns={
                config.date_column: "Date",
                config.sentiment_score_column: "sentiment_score",
            },
            inplace=True,
        )

        if aggregated.empty:
            raise ValueError("No valid data found after processing.")

        return aggregated

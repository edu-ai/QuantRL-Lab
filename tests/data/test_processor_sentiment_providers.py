"""Tests for DataProcessor with new sentiment providers."""

import pandas as pd
import pytest

from quantrl_lab.data.processing import DataProcessor
from quantrl_lab.data.processing.sentiment import HuggingFaceConfig, HuggingFaceProvider, SentimentConfig


class TestDataProcessorWithSentimentProviders:
    """Test DataProcessor with the new sentiment provider pattern."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": range(10),
                "High": range(10, 20),
                "Low": range(0, 10),
                "Close": range(5, 15),
                "Volume": range(1000, 1010),
            }
        )

    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data with sentiment scores."""
        return pd.DataFrame(
            {
                "headline": ["Positive news", "Negative news", "Neutral news"],
                "created_at": pd.date_range("2023-01-01", periods=3, freq="D"),
                "sentiment_score": [0.8, 0.2, 0.5],
            }
        )

    def test_processor_uses_default_hf_provider(self, sample_ohlcv_data, sample_news_data):
        """Test that processor creates default HuggingFace provider when
        news data is present."""
        processor = DataProcessor(sample_ohlcv_data, news_data=sample_news_data)

        assert isinstance(processor.sentiment_provider, HuggingFaceProvider)
        assert processor.sentiment_provider.hf_config.model_name == "ProsusAI/finbert"

    def test_processor_accepts_explicit_provider(self, sample_ohlcv_data):
        """Test that processor accepts explicit sentiment provider."""
        custom_config = HuggingFaceConfig(model_name="custom/model")
        custom_provider = HuggingFaceProvider(custom_config)

        processor = DataProcessor(sample_ohlcv_data, sentiment_provider=custom_provider)

        assert processor.sentiment_provider is custom_provider
        assert processor.sentiment_provider.hf_config.model_name == "custom/model"

    def test_processor_pipeline_with_news_sentiment(self, sample_ohlcv_data, sample_news_data):
        """Test processing pipeline with news sentiment."""
        processor = DataProcessor(sample_ohlcv_data, news_data=sample_news_data)

        result, metadata = processor.data_processing_pipeline()

        # Should have sentiment scores
        assert "sentiment_score" in result.columns
        assert metadata["news_sentiment_applied"] is True

    def test_processor_new_sentiment_config_object(self, sample_ohlcv_data):
        """Test with new SentimentConfig object."""
        config = SentimentConfig(text_column="custom_text", date_column="custom_date")

        processor = DataProcessor(sample_ohlcv_data, sentiment_config=config)

        assert processor.sentiment_config is config
        assert processor.sentiment_config.text_column == "custom_text"


class TestDataProcessorSentimentIntegration:
    """Integration tests for sentiment analysis in data processing
    pipeline."""

    @pytest.fixture
    def sample_data_with_news(self):
        """Create OHLCV and news data."""
        ohlcv = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "Open": range(5),
                "High": range(5, 10),
                "Low": range(0, 5),
                "Close": range(2, 7),
                "Volume": range(1000, 1005),
            }
        )

        news = pd.DataFrame(
            {
                "headline": ["News 1", "News 2", "News 3"],
                "created_at": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "sentiment_score": [0.9, 0.5, 0.1],
            }
        )

        return ohlcv, news

    def test_full_pipeline_with_sentiment_and_indicators(self, sample_data_with_news):
        """Test complete pipeline with sentiment and technical
        indicators."""
        ohlcv, news = sample_data_with_news

        processor = DataProcessor(ohlcv, news_data=news)
        # Note: SMA with default window=20 will create NaN values which get dropped
        # Since we only have 5 rows, all will be dropped. Use smaller window
        result, metadata = processor.data_processing_pipeline(
            indicators=[{"SMA": {"window": 2}}], fillna_strategy="neutral"  # Use window=2 to preserve more data
        )

        # Should have both indicators and sentiment
        # After dropna() we might have fewer rows, but columns should exist
        assert "SMA_2" in result.columns
        assert "sentiment_score" in result.columns
        assert metadata["news_sentiment_applied"] is True
        assert {"SMA": {"window": 2}} in metadata["technical_indicators"]

    def test_pipeline_sentiment_fillna_neutral(self, sample_data_with_news):
        """Test that neutral fillna strategy works."""
        ohlcv, news = sample_data_with_news

        processor = DataProcessor(ohlcv, news_data=news)
        result, metadata = processor.data_processing_pipeline(fillna_strategy="neutral")

        # Days without news should have 0.0 sentiment (neutral)
        assert result["sentiment_score"].notna().all()
        # Should have filled missing dates with 0.0
        zero_scores = result[result["sentiment_score"] == 0.0]
        assert len(zero_scores) > 0

    def test_processor_provides_explicit_provider_instance(self, sample_data_with_news):
        """Test using an explicitly created provider instance."""
        ohlcv, news = sample_data_with_news

        # Create provider with custom config
        hf_config = HuggingFaceConfig(model_name="ProsusAI/finbert")
        provider = HuggingFaceProvider(hf_config)

        processor = DataProcessor(ohlcv, news_data=news, sentiment_provider=provider)
        result, metadata = processor.data_processing_pipeline()

        assert "sentiment_score" in result.columns
        assert metadata["news_sentiment_applied"] is True

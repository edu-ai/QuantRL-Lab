"""Tests for HuggingFaceProvider."""

import pandas as pd
import pytest

from quantrl_lab.data.processing.sentiment import HuggingFaceConfig, HuggingFaceProvider, SentimentConfig


class TestHuggingFaceProviderInit:
    """Test HuggingFaceProvider initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        provider = HuggingFaceProvider()
        assert provider.hf_config is not None
        assert provider.hf_config.model_name == "ProsusAI/finbert"
        assert provider.hf_config.device == -1

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = HuggingFaceConfig(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)
        provider = HuggingFaceProvider(config)
        assert provider.hf_config.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert provider.hf_config.device == 0

    def test_pipeline_lazy_initialization(self):
        """Test that pipeline is not initialized until needed."""
        provider = HuggingFaceProvider()
        assert provider._pipeline is None


class TestHuggingFaceProviderAnalyze:
    """Test HuggingFaceProvider.analyze() method."""

    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data."""
        return pd.DataFrame(
            {
                "headline": [
                    "Company reports strong earnings",
                    "Stock price drops significantly",
                    "Neutral market conditions",
                ],
                "created_at": pd.date_range("2023-01-01", periods=3, freq="D"),
            }
        )

    @pytest.fixture
    def sample_news_with_scores(self):
        """Create sample news data with pre-existing sentiment
        scores."""
        return pd.DataFrame(
            {
                "headline": ["Test headline 1", "Test headline 2"],
                "created_at": pd.date_range("2023-01-01", periods=2, freq="D"),
                "sentiment_score": [0.8, 0.3],
            }
        )

    def test_analyze_with_preexisting_scores(self, sample_news_with_scores):
        """Test that pre-existing scores are used without running
        model."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()

        result = provider.analyze(sample_news_with_scores, config)

        # Should use existing scores
        assert "sentiment_score" in result.columns
        assert "Date" in result.columns
        # Pipeline should not be initialized
        assert provider._pipeline is None

    def test_analyze_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            provider.analyze(empty_df, config)

    def test_analyze_missing_text_column_raises_error(self):
        """Test that missing text column raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig(text_column="headline")

        df = pd.DataFrame(
            {
                "created_at": pd.date_range("2023-01-01", periods=3, freq="D"),
                "other_column": ["text1", "text2", "text3"],
            }
        )

        with pytest.raises(ValueError, match="headline"):
            provider.analyze(df, config)

    def test_analyze_missing_date_column_raises_error(self):
        """Test that missing date column raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig(date_column="created_at")

        df = pd.DataFrame(
            {
                "headline": ["text1", "text2", "text3"],
                "other_date": pd.date_range("2023-01-01", periods=3, freq="D"),
            }
        )

        with pytest.raises(ValueError, match="created_at"):
            provider.analyze(df, config)

    def test_analyze_groups_by_date(self, sample_news_with_scores):
        """Test that analyze aggregates scores by date."""
        # Add multiple entries for same date
        df = pd.DataFrame(
            {
                "headline": ["headline1", "headline2", "headline3"],
                "created_at": ["2023-01-01", "2023-01-01", "2023-01-02"],
                "sentiment_score": [0.8, 0.6, 0.4],
            }
        )

        provider = HuggingFaceProvider()
        config = SentimentConfig()
        result = provider.analyze(df, config)

        # Should have 2 rows (2 unique dates)
        assert len(result) == 2
        # First date should have average of 0.8 and 0.6 = 0.7
        first_row = result[result["Date"] == pd.to_datetime("2023-01-01").date()]
        assert abs(first_row["sentiment_score"].iloc[0] - 0.7) < 0.01

    def test_analyze_returns_correct_columns(self, sample_news_with_scores):
        """Test that analyze returns correctly named columns."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        result = provider.analyze(sample_news_with_scores, config)

        assert "Date" in result.columns
        assert "sentiment_score" in result.columns
        assert len(result.columns) == 2

    def test_analyze_converts_date_to_date_object(self, sample_news_with_scores):
        """Test that date column is converted to date objects."""
        import datetime

        provider = HuggingFaceProvider()
        config = SentimentConfig()
        result = provider.analyze(sample_news_with_scores, config)

        # Date should be date objects, not datetime
        assert result["Date"].dtype == object
        assert isinstance(result["Date"].iloc[0], datetime.date)

    def test_analyze_custom_column_names(self):
        """Test analyze with custom column names."""
        df = pd.DataFrame(
            {
                "text": ["headline1", "headline2"],
                "date": ["2023-01-01", "2023-01-02"],
                "score": [0.8, 0.6],
            }
        )

        provider = HuggingFaceProvider()
        config = SentimentConfig(text_column="text", date_column="date", sentiment_score_column="score")
        result = provider.analyze(df, config)

        assert "Date" in result.columns
        assert "sentiment_score" in result.columns


class TestHuggingFaceProviderProtocol:
    """Test that HuggingFaceProvider implements SentimentProvider
    protocol."""

    def test_implements_protocol(self):
        """Test that provider implements SentimentProvider protocol."""
        from quantrl_lab.data.processing.sentiment import SentimentProvider

        provider = HuggingFaceProvider()
        assert isinstance(provider, SentimentProvider)

    def test_has_analyze_method(self):
        """Test that provider has analyze() method."""
        provider = HuggingFaceProvider()
        assert hasattr(provider, "analyze")
        assert callable(provider.analyze)


class TestHuggingFaceConfig:
    """Test HuggingFaceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HuggingFaceConfig()
        assert config.model_name == "ProsusAI/finbert"
        assert config.device == -1
        assert config.truncation is True
        assert config.top_k == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HuggingFaceConfig(model_name="custom/model", device=0, max_length=256, truncation=False, top_k=3)
        assert config.model_name == "custom/model"
        assert config.device == 0
        assert config.max_length == 256
        assert config.truncation is False
        assert config.top_k == 3

    def test_invalid_device_raises_error(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be -1"):
            HuggingFaceConfig(device=-2)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = HuggingFaceConfig(model_name="test/model", device=0)
        config_dict = config.to_dict()

        assert config_dict["model_name"] == "test/model"
        assert config_dict["device"] == 0
        assert "max_length" in config_dict
        assert "truncation" in config_dict
        assert "top_k" in config_dict

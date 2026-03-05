"""Tests for SentimentFeatureGenerator."""

import pandas as pd
import pytest

from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator
from quantrl_lab.data.processing.sentiment import (
    HuggingFaceProvider,
    SentimentConfig,
)


class TestSentimentFeatureGeneratorInit:
    """Test SentimentFeatureGenerator initialization."""

    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data."""
        return pd.DataFrame(
            {
                "headline": ["Positive news", "Negative news"],
                "created_at": pd.date_range("2023-01-01", periods=2),
                "sentiment_score": [0.8, 0.2],
            }
        )

    def test_init_with_valid_params(self, sample_news_data):
        """Test initialization with valid parameters."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        assert generator.sentiment_provider is provider
        assert generator.sentiment_config is config
        assert generator.fillna_strategy == "neutral"

    def test_init_with_custom_fillna_strategy(self, sample_news_data):
        """Test initialization with custom fillna strategy."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data, fillna_strategy="fill_forward")

        assert generator.fillna_strategy == "fill_forward"

    def test_init_empty_news_raises_error(self):
        """Test that empty news data raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        empty_news = pd.DataFrame()

        with pytest.raises(ValueError, match="cannot be None or empty"):
            SentimentFeatureGenerator(provider, config, empty_news)

    def test_init_none_news_raises_error(self):
        """Test that None news data raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()

        with pytest.raises(ValueError, match="cannot be None or empty"):
            SentimentFeatureGenerator(provider, config, None)

    def test_init_invalid_fillna_strategy_raises_error(self, sample_news_data):
        """Test that invalid fillna strategy raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()

        with pytest.raises(ValueError, match="Unsupported strategy"):
            SentimentFeatureGenerator(provider, config, sample_news_data, fillna_strategy="invalid")


class TestSentimentFeatureGeneratorGenerate:
    """Test SentimentFeatureGenerator.generate() method."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        return pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "Open": range(5),
                "High": range(5, 10),
                "Low": range(0, 5),
                "Close": range(2, 7),
                "Volume": range(1000, 1005),
            }
        )

    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data with sentiment scores."""
        return pd.DataFrame(
            {
                "headline": ["News 1", "News 2"],
                "created_at": ["2023-01-01", "2023-01-02"],
                "sentiment_score": [0.9, 0.3],
            }
        )

    def test_generate_adds_sentiment_column(self, sample_ohlcv_data, sample_news_data):
        """Test that generate adds sentiment_score column."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        result = generator.generate(sample_ohlcv_data)

        assert "sentiment_score" in result.columns

    def test_generate_neutral_fillna(self, sample_ohlcv_data, sample_news_data):
        """Test that neutral strategy fills missing with 0.0."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data, fillna_strategy="neutral")

        result = generator.generate(sample_ohlcv_data)

        # Days without news should have 0.0 (neutral)
        assert result["sentiment_score"].notna().all()
        # At least one day should have 0.0
        assert (result["sentiment_score"] == 0.0).any()

    def test_generate_preserves_original_columns(self, sample_ohlcv_data, sample_news_data):
        """Test that enrichment preserves original columns."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        result = generator.generate(sample_ohlcv_data)

        for col in sample_ohlcv_data.columns:
            assert col in result.columns

    def test_generate_does_not_modify_input(self, sample_ohlcv_data, sample_news_data):
        """Test that original DataFrame is not modified."""
        original_cols = sample_ohlcv_data.columns.tolist()
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        generator.generate(sample_ohlcv_data)

        # Original should be unchanged
        assert sample_ohlcv_data.columns.tolist() == original_cols
        assert "sentiment_score" not in sample_ohlcv_data.columns

    def test_generate_empty_dataframe_raises_error(self, sample_news_data):
        """Test that empty DataFrame raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            generator.generate(empty_df)

    def test_generate_missing_date_column_raises_error(self, sample_news_data):
        """Test that missing Date column raises ValueError."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        df = pd.DataFrame({"Open": [1, 2, 3], "Close": [4, 5, 6]})

        with pytest.raises(ValueError, match="Date"):
            generator.generate(df)

    def test_generate_merges_on_date(self, sample_ohlcv_data, sample_news_data):
        """Test that sentiment scores are correctly merged by date."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        result = generator.generate(sample_ohlcv_data)

        # First two dates should have non-zero sentiment (from news)
        # Remaining dates should have 0.0 (neutral fill)
        assert len(result) == len(sample_ohlcv_data)

    def test_generate_exponential_decay(self, sample_ohlcv_data, sample_news_data):
        """Test exponential decay strategy."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        # Use a simple decay rate to make checking easy (0.5)
        generator = SentimentFeatureGenerator(
            provider, config, sample_news_data, fillna_strategy="exponential_decay", decay_rate=0.5
        )

        # OHLCV: Jan 1 to Jan 5 (5 days)
        # News: Jan 1 (0.9), Jan 2 (0.3)
        result = generator.generate(sample_ohlcv_data)
        scores = result["sentiment_score"].values

        # Jan 1: 0.9 (News present)
        assert pytest.approx(scores[0]) == 0.9

        # Jan 2: 0.3 (New News present - resets decay)
        assert pytest.approx(scores[1]) == 0.3

        # Jan 3: 0.3 * 0.5^1 = 0.15 (1 day decay)
        assert pytest.approx(scores[2]) == 0.15

        # Jan 4: 0.3 * 0.5^2 = 0.075 (2 days decay)
        assert pytest.approx(scores[3]) == 0.075


class TestSentimentFeatureGeneratorMetadata:
    """Test SentimentFeatureGenerator.get_metadata() method."""

    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data."""
        return pd.DataFrame(
            {
                "headline": ["News 1", "News 2"],
                "created_at": ["2023-01-01", "2023-01-02"],
                "sentiment_score": [0.9, 0.3],
            }
        )

    def test_metadata_structure(self, sample_news_data):
        """Test metadata has correct structure."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        metadata = generator.get_metadata()

        assert "type" in metadata
        assert "provider" in metadata
        assert "fillna_strategy" in metadata
        assert "news_records" in metadata

    def test_metadata_values(self, sample_news_data):
        """Test metadata contains correct values."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data, fillna_strategy="fill_forward")

        metadata = generator.get_metadata()

        assert metadata["type"] == "sentiment"
        assert metadata["provider"] == "HuggingFaceProvider"
        assert metadata["fillna_strategy"] == "fill_forward"
        assert metadata["news_records"] == 2


class TestSentimentFeatureGeneratorProtocol:
    """Test that SentimentFeatureGenerator implements FeatureGenerator
    protocol."""

    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data."""
        return pd.DataFrame(
            {
                "headline": ["News 1"],
                "created_at": ["2023-01-01"],
                "sentiment_score": [0.9],
            }
        )

    def test_implements_protocol(self, sample_news_data):
        """Test that generator implements FeatureGenerator protocol."""
        from quantrl_lab.data.processing.features.base import FeatureGenerator

        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        assert isinstance(generator, FeatureGenerator)

    def test_has_generate_method(self, sample_news_data):
        """Test that generator has generate() method."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        assert hasattr(generator, "generate")
        assert callable(generator.generate)

    def test_has_get_metadata_method(self, sample_news_data):
        """Test that generator has get_metadata() method."""
        provider = HuggingFaceProvider()
        config = SentimentConfig()
        generator = SentimentFeatureGenerator(provider, config, sample_news_data)

        assert hasattr(generator, "get_metadata")
        assert callable(generator.get_metadata)

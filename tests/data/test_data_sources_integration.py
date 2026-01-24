"""
Integration tests for data sources using real API calls.

These tests verify that the data loaders work correctly with actual APIs
and that the API interfaces haven't changed or been deprecated.

To run these tests locally:
    pytest tests/data/test_data_sources_integration.py -v

To skip in CI/CD, these tests are marked with @pytest.mark.integration
and can be excluded with:
    pytest -m "not integration"
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Skip all tests in this module if running in CI
pytestmark = pytest.mark.integration


def has_alpaca_credentials() -> bool:
    """Check if Alpaca API credentials are available."""
    return bool(os.environ.get("ALPACA_API_KEY")) and bool(os.environ.get("ALPACA_SECRET_KEY"))


def has_alpha_vantage_credentials() -> bool:
    """Check if Alpha Vantage API key is available."""
    return bool(os.environ.get("ALPHA_VANTAGE_API_KEY"))


class TestYfinanceIntegration:
    """Integration tests for YfinanceDataloader using real API calls."""

    def test_get_historical_ohlcv_data_single_symbol(self):
        """Test fetching historical data for a single symbol."""
        from quantrl_lab.data.sources.yfinance import YfinanceDataloader

        loader = YfinanceDataloader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None, "YFinance returned None - API may have changed"
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "No data returned"
        assert "Symbol" in result.columns
        assert result["Symbol"].iloc[0] == "AAPL"

        # Check OHLCV columns exist
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_historical_ohlcv_data_multiple_symbols(self):
        """Test fetching historical data for multiple symbols."""
        from quantrl_lab.data.sources.yfinance import YfinanceDataloader

        loader = YfinanceDataloader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = loader.get_historical_ohlcv_data(
            symbols=["AAPL", "MSFT"],
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None, "YFinance returned None - API may have changed"
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "No data returned"

        # Check both symbols are present
        symbols = result["Symbol"].unique()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_fundamental_data(self):
        """Test fetching fundamental data."""
        from quantrl_lab.data.sources.yfinance import YfinanceDataloader

        loader = YfinanceDataloader()
        result = loader.get_fundamental_data("AAPL", frequency="quarterly")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "Symbol" in result.columns
        assert "Date" in result.columns

    def test_data_types_are_correct(self):
        """Test that returned data has correct types."""
        from quantrl_lab.data.sources.yfinance import YfinanceDataloader

        loader = YfinanceDataloader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        result = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None, "YFinance returned None - API may have changed"

        # Check numeric columns are numeric
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col]), f"{col} should be numeric"


@pytest.mark.skipif(not has_alpaca_credentials(), reason="Alpaca API credentials not available")
class TestAlpacaIntegration:
    """Integration tests for AlpacaDataLoader using real API calls."""

    def test_get_historical_ohlcv_data_single_symbol(self):
        """Test fetching historical data for a single symbol."""
        from quantrl_lab.data.sources.alpaca import AlpacaDataLoader

        loader = AlpacaDataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check standardized columns exist
        expected_columns = ["Open", "High", "Low", "Close", "Volume", "Symbol"]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_historical_ohlcv_data_multiple_symbols(self):
        """Test fetching historical data for multiple symbols."""
        from quantrl_lab.data.sources.alpaca import AlpacaDataLoader

        loader = AlpacaDataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = loader.get_historical_ohlcv_data(
            symbols=["AAPL", "MSFT"],
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check both symbols are present
        symbols = result["Symbol"].unique()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_get_latest_quote(self):
        """Test fetching latest quote."""
        from quantrl_lab.data.sources.alpaca import AlpacaDataLoader

        loader = AlpacaDataLoader()
        result = loader.get_latest_quote("AAPL")

        assert result is not None
        assert "AAPL" in result

    def test_get_latest_trade(self):
        """Test fetching latest trade."""
        from quantrl_lab.data.sources.alpaca import AlpacaDataLoader

        loader = AlpacaDataLoader()
        result = loader.get_latest_trade("AAPL")

        assert result is not None
        assert "AAPL" in result

    def test_get_news_data(self):
        """Test fetching news data."""
        from quantrl_lab.data.sources.alpaca import AlpacaDataLoader

        loader = AlpacaDataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        result = loader.get_news_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            limit=10,
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # News might be empty for some periods, so just check it's a DataFrame

    def test_is_connected(self):
        """Test connection status with valid credentials."""
        from quantrl_lab.data.sources.alpaca import AlpacaDataLoader

        loader = AlpacaDataLoader()

        assert loader.is_connected() is True


@pytest.mark.skipif(not has_alpha_vantage_credentials(), reason="Alpha Vantage API key not available")
class TestAlphaVantageIntegration:
    """
    Integration tests for AlphaVantageDataLoader using real API calls.

    Note: Alpha Vantage has strict rate limits (25 calls/day for free tier),
    so these tests may fail if run too frequently. Tests are designed to
    handle rate limit responses gracefully.
    """

    def _is_rate_limited(self, result) -> bool:
        """Check if the result indicates rate limiting."""
        if result is None:
            return False
        if isinstance(result, dict) and "Information" in result:
            return True
        return False

    def test_get_historical_ohlcv_data_daily(self):
        """Test fetching daily historical data."""
        from quantrl_lab.data.sources.alpha_vantage import AlphaVantageDataLoader

        loader = AlphaVantageDataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        result = loader.get_historical_ohlcv_data(
            symbols="IBM",  # IBM is commonly used in Alpha Vantage examples
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # May be empty if rate limited, but should still be a DataFrame
        if len(result) > 0:
            expected_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_columns:
                assert col in result.columns, f"Missing column: {col}"

    def test_get_company_overview(self):
        """Test fetching company overview."""
        from quantrl_lab.data.sources.alpha_vantage import AlphaVantageDataLoader

        loader = AlphaVantageDataLoader()
        result = loader._get_company_overview("IBM")

        # May be None or rate limited
        if result is not None and not self._is_rate_limited(result):
            assert isinstance(result, dict)
            assert "Symbol" in result

    def test_get_real_gdp_data(self):
        """Test fetching real GDP data."""
        from quantrl_lab.data.sources.alpha_vantage import AlphaVantageDataLoader

        loader = AlphaVantageDataLoader()
        result = loader._get_real_gdp_data(interval="annual")

        # May be None or rate limited
        if result is not None and not self._is_rate_limited(result):
            assert isinstance(result, dict)
            assert "data" in result or "name" in result


class TestDataSourceRegistryIntegration:
    """Integration tests for DataSourceRegistry."""

    @pytest.mark.skipif(not has_alpaca_credentials(), reason="Alpaca API credentials not available")
    def test_registry_fetches_data(self):
        """Test that registry correctly delegates to primary source."""
        from quantrl_lab.data.source_registry import DataSourceRegistry

        registry = DataSourceRegistry()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        result = registry.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestIndicatorsWithRealData:
    """Test technical indicators with real market data."""

    def test_indicators_on_real_data(self):
        """Test that indicators work correctly on real market data."""
        from quantrl_lab.data.indicators.registry import IndicatorRegistry
        from quantrl_lab.data.sources.yfinance import YfinanceDataloader

        # Fetch real data
        loader = YfinanceDataloader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        df = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert df is not None, "YFinance returned None - cannot test indicators"
        assert len(df) > 0, "No data returned from YFinance"

        # Apply all indicators
        indicators = ["SMA", "EMA", "RSI", "MACD", "ATR", "BB", "STOCH", "OBV"]
        for indicator in indicators:
            result = IndicatorRegistry.apply(indicator, df)
            assert result is not None, f"Indicator {indicator} returned None"
            assert len(result) == len(df), f"Indicator {indicator} changed DataFrame length"

    def test_rsi_values_are_valid_on_real_data(self):
        """Test RSI produces valid values on real data."""
        from quantrl_lab.data.indicators.registry import IndicatorRegistry
        from quantrl_lab.data.sources.yfinance import YfinanceDataloader

        loader = YfinanceDataloader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        df = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        assert df is not None, "YFinance returned None - cannot test RSI"
        assert len(df) > 0, "No data returned from YFinance"

        result = IndicatorRegistry.apply("RSI", df, window=14)

        # Check RSI values are in valid range
        rsi_values = result["RSI_14"].dropna()
        assert (rsi_values >= 0).all(), "RSI values should be >= 0"
        assert (rsi_values <= 100).all(), "RSI values should be <= 100"

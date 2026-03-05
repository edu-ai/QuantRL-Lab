"""Unit tests for technical indicators and the IndicatorRegistry."""

import numpy as np
import pandas as pd
import pytest

from quantrl_lab.data.indicators.registry import IndicatorRegistry
from quantrl_lab.data.indicators.technical import (
    adx,
    atr,
    bollinger_bands,
    cci,
    ema,
    macd,
    mfi,
    on_balance_volume,
    rsi,
    sma,
    stochastic,
    williams_r,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    Create sample OHLCV data for testing.

    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, Close, Volume columns.
    """
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


@pytest.fixture
def multi_symbol_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample OHLCV data with multiple symbols.

    Args:
        sample_ohlcv_df: Base OHLCV DataFrame.

    Returns:
        pd.DataFrame: DataFrame with multiple symbols.
    """
    df1 = sample_ohlcv_df.copy()
    df1["Symbol"] = "AAPL"

    df2 = sample_ohlcv_df.copy()
    df2["Close"] = df2["Close"] * 1.5
    df2["Symbol"] = "GOOGL"

    return pd.concat([df1, df2], ignore_index=True)


class TestIndicatorRegistry:
    """Tests for the IndicatorRegistry class."""

    def test_list_all_returns_registered_indicators(self):
        """Test that list_all returns all registered indicator names."""
        indicators = IndicatorRegistry.list_all()

        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert "SMA" in indicators
        assert "EMA" in indicators
        assert "RSI" in indicators
        assert "MACD" in indicators
        assert "ATR" in indicators
        assert "BB" in indicators
        assert "STOCH" in indicators
        assert "OBV" in indicators
        assert "WILLR" in indicators
        assert "CCI" in indicators
        assert "MFI" in indicators
        assert "ADX" in indicators

    def test_get_returns_callable(self):
        """Test that get returns a callable function."""
        sma_func = IndicatorRegistry.get("SMA")

        assert callable(sma_func)

    def test_get_raises_keyerror_for_unknown_indicator(self):
        """Test that get raises KeyError for unregistered indicator."""
        with pytest.raises(KeyError, match="Indicator 'UNKNOWN' not registered"):
            IndicatorRegistry.get("UNKNOWN")

    def test_apply_executes_indicator(self, sample_ohlcv_df: pd.DataFrame):
        """Test that apply correctly executes an indicator function."""
        result = IndicatorRegistry.apply("SMA", sample_ohlcv_df, window=10)

        assert "SMA_10" in result.columns
        assert len(result) == len(sample_ohlcv_df)

    def test_register_decorator_works(self):
        """Test that the register decorator correctly registers
        functions."""

        @IndicatorRegistry.register(name="TEST_INDICATOR")
        def test_indicator(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["test"] = 1
            return df

        assert "TEST_INDICATOR" in IndicatorRegistry.list_all()

        # Clean up
        del IndicatorRegistry._indicators["TEST_INDICATOR"]


class TestSMA:
    """Tests for Simple Moving Average indicator."""

    def test_sma_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that SMA adds the expected column."""
        result = sma(sample_ohlcv_df, window=20)

        assert "SMA_20" in result.columns

    def test_sma_custom_window(self, sample_ohlcv_df: pd.DataFrame):
        """Test SMA with custom window size."""
        result = sma(sample_ohlcv_df, window=10)

        assert "SMA_10" in result.columns

    def test_sma_values_are_correct(self, sample_ohlcv_df: pd.DataFrame):
        """Test that SMA values are calculated correctly."""
        window = 5
        result = sma(sample_ohlcv_df, window=window)

        # First (window-1) values should be NaN
        assert result[f"SMA_{window}"].iloc[: window - 1].isna().all()

        # Check a specific value
        expected = sample_ohlcv_df["Close"].iloc[:window].mean()
        assert np.isclose(result[f"SMA_{window}"].iloc[window - 1], expected)

    def test_sma_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test SMA with multiple symbols."""
        result = sma(multi_symbol_df, window=10)

        assert "SMA_10" in result.columns
        # Each symbol should have its own SMA calculated independently
        assert not result["SMA_10"].isna().all()

    def test_sma_preserves_original_columns(self, sample_ohlcv_df: pd.DataFrame):
        """Test that SMA preserves original DataFrame columns."""
        original_columns = set(sample_ohlcv_df.columns)
        result = sma(sample_ohlcv_df, window=20)

        assert original_columns.issubset(set(result.columns))


class TestEMA:
    """Tests for Exponential Moving Average indicator."""

    def test_ema_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that EMA adds the expected column."""
        result = ema(sample_ohlcv_df, window=20)

        assert "EMA_20" in result.columns

    def test_ema_custom_window(self, sample_ohlcv_df: pd.DataFrame):
        """Test EMA with custom window size."""
        result = ema(sample_ohlcv_df, window=12)

        assert "EMA_12" in result.columns

    def test_ema_no_nan_values(self, sample_ohlcv_df: pd.DataFrame):
        """Test that EMA has no NaN values (EWM doesn't require
        warmup)."""
        result = ema(sample_ohlcv_df, window=20)

        # EMA should have values from the start
        assert not result["EMA_20"].isna().all()

    def test_ema_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test EMA with multiple symbols."""
        result = ema(multi_symbol_df, window=10)

        assert "EMA_10" in result.columns


class TestRSI:
    """Tests for Relative Strength Index indicator."""

    def test_rsi_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that RSI adds the expected column."""
        result = rsi(sample_ohlcv_df, window=14)

        assert "RSI_14" in result.columns

    def test_rsi_values_in_range(self, sample_ohlcv_df: pd.DataFrame):
        """Test that RSI values are between 0 and 100."""
        result = rsi(sample_ohlcv_df, window=14)
        valid_rsi = result["RSI_14"].dropna()

        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_warmup_period(self, sample_ohlcv_df: pd.DataFrame):
        """Test that RSI has NaN values during warmup period."""
        window = 14
        result = rsi(sample_ohlcv_df, window=window)

        # First window values should be NaN
        assert result["RSI_14"].iloc[:window].isna().all()

    def test_rsi_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test RSI with multiple symbols."""
        result = rsi(multi_symbol_df, window=14)

        assert "RSI_14" in result.columns


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_adds_columns(self, sample_ohlcv_df: pd.DataFrame):
        """Test that MACD adds the expected columns."""
        result = macd(sample_ohlcv_df, fast=12, slow=26, signal=9)

        assert "MACD_line_12_26" in result.columns
        assert "MACD_signal_9" in result.columns

    def test_macd_custom_params(self, sample_ohlcv_df: pd.DataFrame):
        """Test MACD with custom parameters."""
        result = macd(sample_ohlcv_df, fast=5, slow=10, signal=3)

        assert "MACD_line_5_10" in result.columns
        assert "MACD_signal_3" in result.columns

    def test_macd_line_is_difference_of_emas(self, sample_ohlcv_df: pd.DataFrame):
        """Test that MACD line is the difference between fast and slow
        EMAs."""
        result = macd(sample_ohlcv_df, fast=12, slow=26, signal=9)

        fast_ema = sample_ohlcv_df["Close"].ewm(span=12, adjust=False).mean()
        slow_ema = sample_ohlcv_df["Close"].ewm(span=26, adjust=False).mean()
        expected_macd = fast_ema - slow_ema

        np.testing.assert_array_almost_equal(result["MACD_line_12_26"].values, expected_macd.values)

    def test_macd_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test MACD with multiple symbols."""
        result = macd(multi_symbol_df, fast=12, slow=26, signal=9)

        assert "MACD_line_12_26" in result.columns
        assert "MACD_signal_9" in result.columns


class TestATR:
    """Tests for Average True Range indicator."""

    def test_atr_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that ATR adds the expected column."""
        result = atr(sample_ohlcv_df, window=14)

        assert "ATR_14" in result.columns

    def test_atr_values_are_positive(self, sample_ohlcv_df: pd.DataFrame):
        """Test that ATR values are positive."""
        result = atr(sample_ohlcv_df, window=14)
        valid_atr = result["ATR_14"].dropna()

        assert (valid_atr >= 0).all()

    def test_atr_warmup_period(self, sample_ohlcv_df: pd.DataFrame):
        """Test that ATR has NaN values during warmup period."""
        window = 14
        result = atr(sample_ohlcv_df, window=window)

        # First (window-1) values should be NaN
        assert result["ATR_14"].iloc[: window - 1].isna().all()

    def test_atr_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test ATR with multiple symbols."""
        result = atr(multi_symbol_df, window=14)

        assert "ATR_14" in result.columns


class TestBollingerBands:
    """Tests for Bollinger Bands indicator."""

    def test_bollinger_bands_adds_columns(self, sample_ohlcv_df: pd.DataFrame):
        """Test that Bollinger Bands adds the expected columns."""
        result = bollinger_bands(sample_ohlcv_df, window=20, num_std=2.0)

        assert "BB_middle_20" in result.columns
        assert "BB_upper_20_2.0" in result.columns
        assert "BB_lower_20_2.0" in result.columns
        assert "BB_bandwidth_20" in result.columns

    def test_bollinger_bands_upper_greater_than_lower(self, sample_ohlcv_df: pd.DataFrame):
        """Test that upper band is always greater than lower band."""
        result = bollinger_bands(sample_ohlcv_df, window=20, num_std=2.0)

        valid_idx = result["BB_upper_20_2.0"].notna()
        assert (result.loc[valid_idx, "BB_upper_20_2.0"] > result.loc[valid_idx, "BB_lower_20_2.0"]).all()

    def test_bollinger_bands_middle_is_sma(self, sample_ohlcv_df: pd.DataFrame):
        """Test that middle band is the SMA."""
        window = 20
        result = bollinger_bands(sample_ohlcv_df, window=window)

        expected_sma = sample_ohlcv_df["Close"].rolling(window=window).mean()
        np.testing.assert_array_almost_equal(result["BB_middle_20"].values, expected_sma.values)

    def test_bollinger_bands_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test Bollinger Bands with multiple symbols."""
        result = bollinger_bands(multi_symbol_df, window=20, num_std=2.0)

        assert "BB_middle_20" in result.columns
        assert "BB_upper_20_2.0" in result.columns


class TestStochastic:
    """Tests for Stochastic Oscillator indicator."""

    def test_stochastic_adds_columns(self, sample_ohlcv_df: pd.DataFrame):
        """Test that Stochastic adds the expected columns."""
        result = stochastic(sample_ohlcv_df, k_window=14, d_window=3)

        assert "STOCH_%K_14_1" in result.columns
        assert "STOCH_%D_3" in result.columns

    def test_stochastic_values_in_range(self, sample_ohlcv_df: pd.DataFrame):
        """Test that Stochastic values are between 0 and 100."""
        result = stochastic(sample_ohlcv_df, k_window=14, d_window=3)

        valid_k = result["STOCH_%K_14_1"].dropna()
        valid_d = result["STOCH_%D_3"].dropna()

        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_stochastic_with_smoothing(self, sample_ohlcv_df: pd.DataFrame):
        """Test Stochastic with smoothing applied to %K."""
        result = stochastic(sample_ohlcv_df, k_window=14, d_window=3, smooth_k=3)

        assert "STOCH_%K_14_3" in result.columns

    def test_stochastic_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test Stochastic with multiple symbols."""
        result = stochastic(multi_symbol_df, k_window=14, d_window=3)

        assert "STOCH_%K_14_1" in result.columns
        assert "STOCH_%D_3" in result.columns


class TestOBV:
    """Tests for On-Balance Volume indicator."""

    def test_obv_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that OBV adds the expected column."""
        result = on_balance_volume(sample_ohlcv_df)

        assert "OBV" in result.columns

    def test_obv_first_value_equals_first_volume(self, sample_ohlcv_df: pd.DataFrame):
        """Test that first OBV value equals first volume."""
        result = on_balance_volume(sample_ohlcv_df)

        assert result["OBV"].iloc[0] == sample_ohlcv_df["Volume"].iloc[0]

    def test_obv_increases_on_up_day(self):
        """Test that OBV increases when price goes up."""
        df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Volume": [1000, 1000, 1000],
            }
        )
        result = on_balance_volume(df)

        assert result["OBV"].iloc[1] > result["OBV"].iloc[0]
        assert result["OBV"].iloc[2] > result["OBV"].iloc[1]

    def test_obv_decreases_on_down_day(self):
        """Test that OBV decreases when price goes down."""
        df = pd.DataFrame(
            {
                "Close": [100, 99, 98],
                "Volume": [1000, 1000, 1000],
            }
        )
        result = on_balance_volume(df)

        assert result["OBV"].iloc[1] < result["OBV"].iloc[0]
        assert result["OBV"].iloc[2] < result["OBV"].iloc[1]

    def test_obv_unchanged_on_flat_day(self):
        """Test that OBV stays the same when price doesn't change."""
        df = pd.DataFrame(
            {
                "Close": [100, 100, 100],
                "Volume": [1000, 1000, 1000],
            }
        )
        result = on_balance_volume(df)

        assert result["OBV"].iloc[1] == result["OBV"].iloc[0]
        assert result["OBV"].iloc[2] == result["OBV"].iloc[1]

    def test_obv_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test OBV with multiple symbols."""
        result = on_balance_volume(multi_symbol_df)

        assert "OBV" in result.columns


class TestWilliamsR:
    """Tests for Williams %R indicator."""

    def test_willr_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that Williams %R adds the expected column."""
        result = williams_r(sample_ohlcv_df, window=14)
        assert "WILLR_14" in result.columns

    def test_willr_values_in_range(self, sample_ohlcv_df: pd.DataFrame):
        """Test that Williams %R values are between -100 and 0."""
        result = williams_r(sample_ohlcv_df, window=14)
        valid_willr = result["WILLR_14"].dropna()

        assert (valid_willr >= -100).all()
        assert (valid_willr <= 0).all()

    def test_willr_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test Williams %R with multiple symbols."""
        result = williams_r(multi_symbol_df, window=14)
        assert "WILLR_14" in result.columns


class TestCCI:
    """Tests for Commodity Channel Index indicator."""

    def test_cci_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that CCI adds the expected column."""
        result = cci(sample_ohlcv_df, window=20)
        assert "CCI_20" in result.columns

    def test_cci_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test CCI with multiple symbols."""
        result = cci(multi_symbol_df, window=20)
        assert "CCI_20" in result.columns


class TestMFI:
    """Tests for Money Flow Index indicator."""

    def test_mfi_adds_column(self, sample_ohlcv_df: pd.DataFrame):
        """Test that MFI adds the expected column."""
        result = mfi(sample_ohlcv_df, window=14)
        assert "MFI_14" in result.columns

    def test_mfi_values_in_range(self, sample_ohlcv_df: pd.DataFrame):
        """Test that MFI values are between 0 and 100."""
        result = mfi(sample_ohlcv_df, window=14)
        valid_mfi = result["MFI_14"].dropna()

        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()

    def test_mfi_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test MFI with multiple symbols."""
        result = mfi(multi_symbol_df, window=14)
        assert "MFI_14" in result.columns


class TestADX:
    """Tests for Average Directional Index indicator."""

    def test_adx_adds_columns(self, sample_ohlcv_df: pd.DataFrame):
        """Test that ADX adds the expected columns."""
        result = adx(sample_ohlcv_df, window=14)
        assert "ADX_14" in result.columns
        assert "ADX_pos_14" in result.columns
        assert "ADX_neg_14" in result.columns

    def test_adx_values_in_range(self, sample_ohlcv_df: pd.DataFrame):
        """Test that ADX values are between 0 and 100."""
        result = adx(sample_ohlcv_df, window=14)
        valid_adx = result["ADX_14"].dropna()

        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_adx_with_multiple_symbols(self, multi_symbol_df: pd.DataFrame):
        """Test ADX with multiple symbols."""
        result = adx(multi_symbol_df, window=14)
        assert "ADX_14" in result.columns
        assert "ADX_pos_14" in result.columns
        assert "ADX_neg_14" in result.columns

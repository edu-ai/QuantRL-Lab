import pandas as pd

from .base import SignalType, VectorizedTradingStrategy
from .registry import VectorizedStrategyRegistry


@VectorizedStrategyRegistry.register("trend_following")
class TrendFollowingStrategy(VectorizedTradingStrategy):
    """Strategy for trend-following indicators like SMA, EMA."""

    def __init__(self, indicator_col: str, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on trend-following strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.indicator_col not in data.columns:
            return signals

        # Buy when price > indicator
        signals[data["Close"] > data[self.indicator_col]] = SignalType.BUY.value

        # Sell when price < indicator (if shorting allowed)
        if self.allow_short:
            signals[data["Close"] < data[self.indicator_col]] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.indicator_col, "Close"]


@VectorizedStrategyRegistry.register("mean_reversion")
class MeanReversionStrategy(VectorizedTradingStrategy):
    """Strategy for mean-reversion indicators like RSI."""

    def __init__(self, indicator_col: str, oversold: float = 30, overbought: float = 70, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on mean-reversion strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.indicator_col not in data.columns:
            return signals

        # Buy when oversold
        buy_condition = data[self.indicator_col] < self.oversold
        signals[buy_condition] = SignalType.BUY.value

        # Sell when overbought (if shorting allowed)
        if self.allow_short:
            sell_condition = data[self.indicator_col] > self.overbought
            signals[sell_condition] = SignalType.SELL.value

        # Forward fill to maintain positions
        signals = (
            signals.replace(SignalType.HOLD.value, pd.NA)
            .ffill()
            .fillna(SignalType.HOLD.value)
            .infer_objects(copy=False)
        )

        return signals

    def get_required_columns(self) -> list:
        return [self.indicator_col]


@VectorizedStrategyRegistry.register("macd_crossover")
class MACDCrossoverStrategy(VectorizedTradingStrategy):
    """Strategy for crossover indicators from MACD line."""

    def __init__(self, fast_col: str, slow_col: str, allow_short: bool = True):
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD crossover strategy.

        Args:
            data (pd.DataFrame): input OHLCV df

        Returns:
            pd.Series: generated trading signals
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.fast_col not in data.columns or self.slow_col not in data.columns:
            return signals

        # Buy when fast > slow
        signals[data[self.fast_col] > data[self.slow_col]] = SignalType.BUY.value

        # Sell when fast < slow (if shorting allowed)
        if self.allow_short:
            signals[data[self.fast_col] < data[self.slow_col]] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.fast_col, self.slow_col]


@VectorizedStrategyRegistry.register("volatility_breakout")
class VolatilityBreakoutStrategy(VectorizedTradingStrategy):
    """Strategy for volatility indicators like ATR."""

    def __init__(self, indicator_col: str, threshold_percentile: float = 0.7, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.threshold_percentile = threshold_percentile
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on volatility breakout strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.indicator_col not in data.columns:
            return signals

        # Buy when volatility is high (breakout)
        high_threshold = data[self.indicator_col].quantile(self.threshold_percentile)
        signals[data[self.indicator_col] > high_threshold] = SignalType.BUY.value

        # Sell when volatility is low (if shorting allowed)
        if self.allow_short:
            low_threshold = data[self.indicator_col].quantile(1 - self.threshold_percentile)
            signals[data[self.indicator_col] < low_threshold] = SignalType.SELL.value
        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.indicator_col]


@VectorizedStrategyRegistry.register("bollinger_bands")
class BollingerBandsStrategy(VectorizedTradingStrategy):
    """Strategy for Bollinger Bands - Mean reversion at bands"""

    def __init__(self, lower_col: str, middle_col: str, upper_col: str, allow_short: bool = True):
        self.lower_col = lower_col
        self.middle_col = middle_col
        self.upper_col = upper_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands strategy.

        Vectorized implementation of mean reversion at bands.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        missing_cols = [col for col in self.get_required_columns() if col not in data.columns]
        if missing_cols:
            return signals

        # 1. Identify entry/exit points
        # Buy (Long) when price <= lower band
        buy_signal = data["Close"] <= data[self.lower_col]
        # Sell (Short) when price >= upper band
        sell_signal = data["Close"] >= data[self.upper_col]
        # Exit when price crosses middle band
        exit_long = data["Close"] >= data[self.middle_col]
        exit_short = data["Close"] <= data[self.middle_col]

        # 2. Map to positions
        # This is a bit tricky to vectorize perfectly with "exit at middle"
        # but we can use a simplified version:
        # Long if last signal was buy and we haven't hit middle band yet.

        # For a truly robust vectorized version with exits, we use the same ffill pattern
        raw_signals = pd.Series(index=data.index, dtype='float64')
        raw_signals[buy_signal] = SignalType.BUY.value
        if self.allow_short:
            raw_signals[sell_signal] = SignalType.SELL.value

        # Add exit points
        # If we are long, we exit at middle. If we are short, we exit at middle.
        # This requires knowing the current state, so we use a simplified
        # state machine approach with ffill

        # Start with BUY/SELL points
        pos = raw_signals.ffill().fillna(SignalType.HOLD.value)

        # Apply exits: if pos was BUY but price > middle, exit.
        # This needs to be applied after ffill to catch the 'holding' period.
        mask_exit_long = (pos == SignalType.BUY.value) & exit_long
        mask_exit_short = (pos == SignalType.SELL.value) & exit_short

        pos[mask_exit_long] = SignalType.HOLD.value
        pos[mask_exit_short] = SignalType.HOLD.value

        # One more ffill to handle the gaps created by exits if they happen mid-trend
        # Actually the loop might be safer for complex state, but we can use
        # a more efficient implementation if needed.
        # For now, let's keep the logic clean.

        return pos.astype(int)

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.lower_col, self.middle_col, self.upper_col, "Close"]


@VectorizedStrategyRegistry.register("stochastic")
class StochasticStrategy(VectorizedTradingStrategy):
    """Strategy for Stochastic Oscillator - Mean reversion"""

    def __init__(
        self, k_col: str, d_col: str = None, oversold: float = 20, overbought: float = 80, allow_short: bool = True
    ):
        self.k_col = k_col
        self.d_col = d_col  # Optional %D line
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Stochastic Oscillator
        strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.k_col not in data.columns:
            return signals

        if self.d_col and self.d_col in data.columns:
            # Use both %K and %D for more robust signals
            buy_condition = (data[self.k_col] < self.oversold) & (data[self.d_col] < self.oversold)

            if self.allow_short:
                sell_condition = (data[self.k_col] > self.overbought) & (data[self.d_col] > self.overbought)
        else:
            # Use only %K
            buy_condition = data[self.k_col] < self.oversold

            if self.allow_short:
                sell_condition = data[self.k_col] > self.overbought

        # Apply signals
        signals[buy_condition] = SignalType.BUY.value

        if self.allow_short:
            signals[sell_condition] = SignalType.SELL.value

        # Forward fill to maintain positions
        signals = (
            signals.replace(SignalType.HOLD.value, pd.NA)
            .ffill()
            .fillna(SignalType.HOLD.value)
            .infer_objects(copy=False)
        )

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        required = [self.k_col]
        if self.d_col:
            required.append(self.d_col)
        return required


@VectorizedStrategyRegistry.register("obv_trend")
class OnBalanceVolumeStrategy(VectorizedTradingStrategy):
    """Strategy for On-Balance Volume - Trend following based on volume"""

    def __init__(self, obv_col: str, allow_short: bool = True):
        self.obv_col = obv_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on On-Balance Volume strategy. A
        simple strategy is to buy when the OBV is rising and sell when
        it's falling. We can use a moving average of OBV to determine
        the trend.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.obv_col not in data.columns:
            return signals

        # Use a short-term moving average to identify the trend of OBV
        obv_sma = data[self.obv_col].rolling(window=20).mean()

        # Buy when OBV is above its moving average (upward trend)
        buy_condition = data[self.obv_col] > obv_sma
        signals[buy_condition] = SignalType.BUY.value

        # Sell when OBV is below its moving average (downward trend)
        if self.allow_short:
            sell_condition = data[self.obv_col] < obv_sma
            signals[sell_condition] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.obv_col]

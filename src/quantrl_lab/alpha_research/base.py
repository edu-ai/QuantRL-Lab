from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class SignalType(Enum):
    # Just 3 possible signals for simplicity
    BUY = 1
    SELL = -1
    HOLD = 0


class VectorizedTradingStrategy(ABC):
    """
    Base strategy class for vectorized trading strategies.

    We will be using the results from vectorized trading strategies to
    decide on the feature selection process.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate discrete trading signals (Buy/Sell/Hold) for the given
        data.

        This is useful for backtesting distinct entry/exit rules.

        Args:
            data (pd.DataFrame): Input market data

        Returns:
            pd.Series: Generated trading signals (1, -1, 0)
        """
        raise NotImplementedError

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores (e.g., -1.0 to 1.0) for the
        given data.

        This is useful for RL observation features and signal analysis (IC).
        By default, it raises NotImplementedError, but subclasses should implement this
        to support advanced signal discovery.

        Args:
            data (pd.DataFrame): Input market data

        Returns:
            pd.Series: Continuous alpha scores
        """
        raise NotImplementedError("This strategy does not support continuous score generation.")

    @staticmethod
    def _rolling_zscore(series: "pd.Series", window: int) -> "pd.Series":
        """Normalize a series to roughly [-1, 1] via rolling Z-score /
        3."""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return ((series - mean) / (std + 1e-9) / 3.0).clip(-1.0, 1.0)

    @abstractmethod
    def get_required_columns(self) -> list:
        """
        Return list of required columns for this strategy.

        Raises:
            NotImplementedError: If not implemented

        Returns:
            list: List of required column names
        """
        raise NotImplementedError

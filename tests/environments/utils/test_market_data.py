import numpy as np
import pandas as pd
import pytest

from quantrl_lab.environments.utils.market_data import (
    auto_detect_price_column,
    calc_trend,
    detect_column_index,
)


class TestMarketData:
    def test_detect_column_index_exact(self):
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        assert detect_column_index(df, ["B"]) == 1
        assert detect_column_index(df, ["A", "B"]) == 0
        assert detect_column_index(df, ["Z", "C"]) == 2

    def test_detect_column_index_case_insensitive(self):
        df = pd.DataFrame({"Open": [1], "high": [2], "LOW": [3]})
        assert detect_column_index(df, ["open"]) == 0
        assert detect_column_index(df, ["High"]) == 1
        assert detect_column_index(df, ["low"]) == 2

    def test_detect_column_index_not_found(self):
        df = pd.DataFrame({"A": [1]})
        assert detect_column_index(df, ["B"]) is None

    def test_auto_detect_price_column_priority(self):
        # Priority: close > price > adj_close
        df = pd.DataFrame({"adj_close": [1], "price": [2], "close": [3]})
        assert auto_detect_price_column(df) == 2  # 'close' is at index 2

        df = pd.DataFrame({"adj_close": [1], "price": [2]})
        assert auto_detect_price_column(df) == 1  # 'price' is at index 1

        df = pd.DataFrame({"adj_close": [1]})
        assert auto_detect_price_column(df) == 0  # 'adj_close' is at index 0

    def test_auto_detect_price_column_fallback(self):
        # Fallback to partial match "close" or "price"
        df = pd.DataFrame({"my_close_price": [1], "volume": [100]})
        assert auto_detect_price_column(df) == 0

    def test_auto_detect_price_column_failure(self):
        df = pd.DataFrame({"volume": [1], "date": [2]})
        with pytest.raises(ValueError, match="Could not auto-detect price column"):
            auto_detect_price_column(df)

    def test_calc_trend_positive(self):
        # Perfectly increasing sequence: 0, 1, 2, 3, 4
        # Slope = 1.0, Max = 4.0 -> Trend = 0.25
        prices = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        trend = calc_trend(prices)
        assert np.isclose(trend, 0.25)

    def test_calc_trend_negative(self):
        # Perfectly decreasing sequence: 4, 3, 2, 1, 0
        # Slope = -1.0, Max = 4.0 -> Trend = -0.25
        prices = np.array([4, 3, 2, 1, 0], dtype=np.float32)
        trend = calc_trend(prices)
        assert np.isclose(trend, -0.25)

    def test_calc_trend_flat(self):
        prices = np.array([10, 10, 10, 10, 10], dtype=np.float32)
        trend = calc_trend(prices)
        # Slope close to 0
        assert np.isclose(trend, 0.0, atol=1e-9)

    def test_calc_trend_insufficient_data(self):
        assert calc_trend(np.array([])) == 0.0
        assert calc_trend(np.array([1.0])) == 0.0

    def test_calc_trend_zero_max(self):
        # Prevent division by zero if max price is 0
        prices = np.array([0, 0, 0], dtype=np.float32)
        assert calc_trend(prices) == 0.0

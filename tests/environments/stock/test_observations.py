from unittest.mock import MagicMock

import numpy as np
import pytest

from quantrl_lab.environments.stock.components.portfolio import StockPortfolio
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy


@pytest.fixture
def mock_env():
    env = MagicMock()

    # Setup standard attributes
    env.window_size = 5
    env.num_features = 3
    env.current_step = 10
    env.original_columns = ["Close", "RSI", "SMA"]
    env.price_column_index = 0

    # Create deterministic data (Step 0 to 19)
    # Col 0 (Close): 100, 101, 102...
    # Col 1 (RSI): 50, 51, 52...
    # Col 2 (SMA): 100, 100.5, 101...
    data = np.zeros((20, 3))
    for i in range(20):
        data[i, 0] = 100 + i  # Price
        data[i, 1] = 50 + i  # RSI
        data[i, 2] = 100 + (i / 2)  # SMA
    env.data = data

    # Mock portfolio
    env.portfolio = MagicMock(spec=StockPortfolio)
    env.portfolio.total_shares = 0
    env.portfolio.balance = 10000.0
    env.portfolio.initial_balance = 10000.0
    env.portfolio.get_value.return_value = 10000.0
    env.portfolio.executed_orders_history = []
    env.portfolio.stop_loss_orders = []
    env.portfolio.take_profit_orders = []

    # Mock helper methods
    env._get_current_price.return_value = 110.0  # Price at step 10

    return env


class TestFeatureAwareObservation:
    def test_shape_definition(self, mock_env):
        strategy = FeatureAwareObservationStrategy()
        space = strategy.define_observation_space(mock_env)

        # Expected: (Window=5 * Features=3) + Portfolio=9 = 24
        assert space.shape == (24,)
        assert space.dtype == np.float32

    def test_smart_normalization(self, mock_env):
        strategy = FeatureAwareObservationStrategy()
        obs = strategy.build_observation(mock_env)

        # Extract the window part (first 15 elements)
        window_part = obs[:15].reshape(5, 3)

        # Window range: [Step 6, 7, 8, 9, 10] (current_step=10, window=5)
        # Note: In the code, start_idx = 10 - 5 + 1 = 6. End = 11.
        # Wait, let's check code: start_idx = max(0, env.current_step - env.window_size + 1)
        # 10 - 5 + 1 = 6. end_idx = 10 + 1 = 11.
        # Indices: 6, 7, 8, 9, 10.

        # --- Check Price Column (Col 0, "Close") ---
        # Raw Values: 106, 107, 108, 109, 110
        # Normalization: relative to FIRST step in window (106)
        # Expected: 1.0, 1.0094, 1.0188, 1.0283, 1.0377
        assert window_part[0, 0] == pytest.approx(1.0)
        assert window_part[4, 0] == pytest.approx(110 / 106)

        # --- Check Stationary Column (Col 1, "RSI") ---
        # Raw Values: 56, 57, 58, 59, 60
        # Normalization: Divided by 100 (because it's RSI)
        # Expected: 0.56, 0.57...
        assert window_part[0, 1] == pytest.approx(0.56)
        assert window_part[4, 1] == pytest.approx(0.60)

        # --- Check SMA Column (Col 2, "SMA") ---
        # Should be treated as price-like
        # Raw at step 6: 100 + 3 = 103
        # Raw at step 10: 100 + 5 = 105
        # Expected: 1.0, ..., 105/103
        assert window_part[0, 2] == pytest.approx(1.0)
        assert window_part[4, 2] == pytest.approx(105 / 103)

    def test_portfolio_features_integration(self, mock_env):
        # Update mock to simulate holding a position
        mock_env.portfolio.total_shares = 10
        mock_env._get_current_price.return_value = 110.0
        mock_env.portfolio.get_value.return_value = 11100.0  # 10*110 + cash

        # Mock entry price history
        mock_env.portfolio.executed_orders_history = [
            {"type": "market_buy", "price": 100.0},
            {"type": "limit_buy_executed", "price": 102.0},
        ]
        # Avg entry = 101.0

        strategy = FeatureAwareObservationStrategy()
        obs = strategy.build_observation(mock_env)

        portfolio_features = obs[-9:]  # Last 9 elements

        # 2. Position Size Ratio = (10 * 110) / 11100 = 1100 / 11100 ~= 0.099
        assert portfolio_features[1] == pytest.approx(1100 / 11100)

        # 3. Unrealized PL % = (110 - 101) / 101 = 9/101 ~= 0.089
        assert portfolio_features[2] == pytest.approx(9 / 101)

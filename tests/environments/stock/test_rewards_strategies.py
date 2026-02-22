from unittest.mock import MagicMock

import pytest

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy, TradingEnvProtocol
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.expiration import OrderExpirationPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty
from quantrl_lab.environments.stock.strategies.rewards.sharpe import DifferentialSharpeReward
from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward


class MockReward(BaseRewardStrategy):
    def __init__(self, value):
        self.value = value

    def calculate_reward(self, env):
        return self.value


def test_composite_reward_calculation():
    """Test that CompositeReward correctly weights its components."""
    # Setup 3 sub-rewards
    r1 = MockReward(1.0)
    r2 = MockReward(0.5)
    r3 = MockReward(-0.2)

    # Create composite with weights
    composite = CompositeReward(strategies=[r1, r2, r3], weights=[0.5, 0.3, 0.2])

    # Expected: (1.0 * 0.5) + (0.5 * 0.3) + (-0.2 * 0.2)
    # = 0.5 + 0.15 - 0.04 = 0.61

    env = MagicMock(spec=TradingEnvProtocol)
    reward = composite.calculate_reward(env)
    assert reward == pytest.approx(0.61)


def test_invalid_action_penalty():
    """Test that InvalidActionPenalty returns penalty only on invalid
    actions."""
    env = MagicMock(spec=TradingEnvProtocol)
    strategy = InvalidActionPenalty(penalty=-5.0)

    # Case 1: Valid Action
    env.decoded_action_info = {"invalid_action_attempt": False}
    assert strategy.calculate_reward(env) == 0.0

    # Case 2: Invalid Action
    env.decoded_action_info = {"invalid_action_attempt": True}
    assert strategy.calculate_reward(env) == -5.0


def test_turnover_penalty_reward():
    """Test that TurnoverPenaltyReward correctly calculates penalty
    based on fees."""
    env = MagicMock()
    env.current_step = 10
    env.portfolio.transaction_cost_pct = 0.001  # 0.1%

    # Strategy with 2x penalty multiplier
    strategy = TurnoverPenaltyReward(penalty_factor=2.0)

    # Case 1: No trades in this step
    env.portfolio.executed_orders_history = []
    assert strategy.calculate_reward(env) == 0.0

    # Case 2: Trades in PREVIOUS step (should be ignored)
    env.portfolio.executed_orders_history = [{"step": 9, "type": "market_buy", "price": 100.0, "shares": 10}]
    assert strategy.calculate_reward(env) == 0.0

    # Case 3: Trades in CURRENT step
    # Buy: 100 shares @ $10. Value = $1000. Fee = $1.
    # Sell: 50 shares @ $20. Value = $1000. Fee = $1.
    # Total Fee = $2. Penalty = $2 * 2.0 = -4.0.
    env.portfolio.executed_orders_history = [
        {"step": 9, "type": "market_buy", "price": 100.0, "shares": 10},  # Old
        {"step": 10, "type": "market_buy", "price": 10.0, "shares": 100},  # New
        {"step": 10, "type": "market_sell", "price": 20.0, "shares": 50},  # New
    ]

    reward = strategy.calculate_reward(env)

    # Calculation verification
    # Buy Value: 10 * 100 = 1000
    # Sell Value: 20 * 50 = 1000
    # Total Value: 2000
    # Fees: 2000 * 0.001 = 2.0
    # Penalty: 2.0 * 2.0 = 4.0 (negative)

    assert reward == pytest.approx(-4.0)


def test_order_expiration_penalty_reward():
    """Test that OrderExpirationPenaltyReward counts expired orders
    correctly."""
    env = MagicMock()
    env.current_step = 10

    # Strategy with -0.5 penalty per expiration
    strategy = OrderExpirationPenaltyReward(penalty_per_order=-0.5)

    # Case 1: No expirations
    env.portfolio.executed_orders_history = [{"step": 10, "type": "market_buy", "price": 100.0}]
    assert strategy.calculate_reward(env) == 0.0

    # Case 2: Expirations in CURRENT step
    env.portfolio.executed_orders_history = [
        {"step": 9, "type": "limit_buy_expired"},  # Old
        {"step": 10, "type": "limit_buy_expired"},  # Count 1
        {"step": 10, "type": "stop_loss_expired"},  # Count 2
        {"step": 10, "type": "market_buy"},  # Ignored
    ]

    reward = strategy.calculate_reward(env)
    assert reward == -1.0  # 2 expirations * -0.5


def test_differential_sharpe_reward():
    """Test DifferentialSharpeReward calculations."""
    env = MagicMock()
    # Use fast decay for easier manual calculation tracking
    strategy = DifferentialSharpeReward(risk_free_rate=0.0, decay=0.5)

    # Step 1: 10% Return
    # Mean = 0.1, MeanSq = 0.01, Var = 0, Std = epsilon
    # Reward ~ 0.1 / epsilon -> Positive large number
    env.prev_portfolio_value = 100.0
    env.portfolio.get_value.return_value = 110.0  # +10%
    r1 = strategy.calculate_reward(env)
    assert r1 > 0

    # Step 2: 0% Return
    # New Mean = 0.5(0.1) + 0.5(0.0) = 0.05
    # New MeanSq = 0.5(0.01) + 0.5(0.0) = 0.005
    # Var = 0.005 - 0.05^2 = 0.005 - 0.0025 = 0.0025
    # Std = sqrt(0.0025) = 0.05
    # Reward = 0.0 / 0.05 = 0.0
    env.prev_portfolio_value = 110.0
    env.portfolio.get_value.return_value = 110.0  # +0%
    r2 = strategy.calculate_reward(env)
    assert r2 == pytest.approx(0.0, abs=1e-5)


def test_drawdown_penalty_reward():
    """Test DrawdownPenaltyReward tracking and penalty."""
    env = MagicMock()
    strategy = DrawdownPenaltyReward(penalty_factor=1.0)

    # Step 1: Value 100. Max=100. Drawdown=0. Reward=0.
    env._get_current_price.return_value = 10.0
    env.portfolio.get_value.return_value = 100.0
    assert strategy.calculate_reward(env) == 0.0

    # Step 2: Value 90. Max=100. Drawdown=10% (0.1). Penalty=-0.1.
    env.portfolio.get_value.return_value = 90.0
    assert strategy.calculate_reward(env) == pytest.approx(-0.1)

    # Step 3: Value 95. Max=100. Drawdown=5% (0.05). Penalty=-0.05.
    env.portfolio.get_value.return_value = 95.0
    assert strategy.calculate_reward(env) == pytest.approx(-0.05)

    # Step 4: Value 110. Max=110. Drawdown=0. Reward=0.
    env.portfolio.get_value.return_value = 110.0
    assert strategy.calculate_reward(env) == 0.0

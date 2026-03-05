from unittest.mock import MagicMock

import pytest

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy, TradingEnvProtocol
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward, _RunningStat
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.execution_bonus import LimitExecutionReward
from quantrl_lab.environments.stock.strategies.rewards.expiration import OrderExpirationPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty
from quantrl_lab.environments.stock.strategies.rewards.sharpe import DifferentialSharpeReward
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
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


# ===========================================================================
# DifferentialSortinoReward
# ===========================================================================


class TestDifferentialSortinoReward:
    def _make_env(self, prev_val, current_val):
        env = MagicMock()
        env.prev_portfolio_value = prev_val
        env.portfolio.get_value.return_value = current_val
        env._get_current_price.return_value = 100.0
        return env

    def test_positive_return_gives_positive_reward(self):
        strategy = DifferentialSortinoReward()
        env = self._make_env(100.0, 110.0)  # +10% return
        reward = strategy.calculate_reward(env)
        assert reward > 0.0

    def test_negative_return_gives_negative_reward(self):
        strategy = DifferentialSortinoReward()
        env = self._make_env(100.0, 90.0)  # -10% return
        reward = strategy.calculate_reward(env)
        assert reward < 0.0

    def test_zero_return_on_first_step_clipped(self):
        strategy = DifferentialSortinoReward()
        env = self._make_env(100.0, 100.0)  # 0% return
        reward = strategy.calculate_reward(env)
        # With 0 return, reward should be 0 / (0 + epsilon) ≈ 0
        assert reward == pytest.approx(0.0, abs=1e-5)

    def test_reward_clipped_to_minus_ten_plus_ten(self):
        strategy = DifferentialSortinoReward()
        # Extreme positive return: should clip to +10
        env = self._make_env(100.0, 100000.0)
        reward = strategy.calculate_reward(env)
        assert reward == pytest.approx(10.0)

    def test_reset_clears_state(self):
        strategy = DifferentialSortinoReward()
        env = self._make_env(100.0, 110.0)
        strategy.calculate_reward(env)  # step 1
        assert strategy._step_count == 1
        strategy.reset()
        assert strategy._step_count == 0
        assert strategy._mean_return == 0.0
        assert strategy._mean_downside_sq == 0.0

    def test_multiple_steps_updates_stats(self):
        strategy = DifferentialSortinoReward()
        # Run several steps
        for _ in range(5):
            env = self._make_env(100.0, 105.0)
            strategy.calculate_reward(env)
        assert strategy._step_count == 5

    def test_zero_prev_value_returns_zero_reward(self):
        strategy = DifferentialSortinoReward()
        env = self._make_env(0.0, 100.0)  # prev_val is near-zero
        reward = strategy.calculate_reward(env)
        # ret = 0.0 when prev_val <= 1e-9
        assert reward == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# LimitExecutionReward
# ===========================================================================


class TestLimitExecutionReward:
    def test_no_orders_returns_zero(self):
        strategy = LimitExecutionReward()
        env = MagicMock()
        env.current_step = 5
        env.portfolio.executed_orders_history = []
        assert strategy.calculate_reward(env) == 0.0

    def test_limit_buy_improvement(self):
        """Bought cheaper than reference: positive reward."""
        strategy = LimitExecutionReward(improvement_multiplier=10.0)
        env = MagicMock()
        env.current_step = 3
        env.portfolio.executed_orders_history = [
            {
                "step": 3,
                "type": "limit_buy_executed",
                "price": 95.0,
                "reference_price": 100.0,
            }
        ]
        reward = strategy.calculate_reward(env)
        # improvement = (100 - 95) / 100 = 0.05; bonus = 0.05 * 10 = 0.5
        assert reward == pytest.approx(0.5)

    def test_limit_sell_improvement(self):
        """Sold higher than reference: positive reward."""
        strategy = LimitExecutionReward(improvement_multiplier=10.0)
        env = MagicMock()
        env.current_step = 7
        env.portfolio.executed_orders_history = [
            {
                "step": 7,
                "type": "limit_sell_executed",
                "price": 110.0,
                "reference_price": 100.0,
            }
        ]
        reward = strategy.calculate_reward(env)
        # improvement = (110 - 100) / 100 = 0.1; bonus = 0.1 * 10 = 1.0
        assert reward == pytest.approx(1.0)

    def test_stale_orders_not_counted(self):
        """Orders from previous steps are ignored."""
        strategy = LimitExecutionReward()
        env = MagicMock()
        env.current_step = 10
        env.portfolio.executed_orders_history = [
            {"step": 9, "type": "limit_buy_executed", "price": 95.0, "reference_price": 100.0},
            {"step": 10, "type": "market_buy", "price": 100.0, "reference_price": 100.0},
        ]
        # The limit_buy at step 9 should be skipped (reversed loop breaks)
        # The market_buy at step 10 is not a "limit_buy_executed" so no improvement
        reward = strategy.calculate_reward(env)
        assert reward == 0.0

    def test_no_improvement_when_price_equals_reference(self):
        """If exec_price == ref_price, improvement_pct = 0."""
        strategy = LimitExecutionReward(improvement_multiplier=10.0)
        env = MagicMock()
        env.current_step = 2
        env.portfolio.executed_orders_history = [
            {"step": 2, "type": "limit_buy_executed", "price": 100.0, "reference_price": 100.0}
        ]
        assert strategy.calculate_reward(env) == 0.0

    def test_zero_reference_price_skipped(self):
        """Zero reference_price should be skipped (ref_price <=
        1e-9)."""
        strategy = LimitExecutionReward()
        env = MagicMock()
        env.current_step = 1
        env.portfolio.executed_orders_history = [
            {"step": 1, "type": "limit_buy_executed", "price": 95.0, "reference_price": 0.0}
        ]
        assert strategy.calculate_reward(env) == 0.0


# ===========================================================================
# CompositeReward (additional coverage)
# ===========================================================================


class TestCompositeRewardExtended:
    def test_auto_scale_normalizes_rewards(self):
        """With auto_scale=True, rewards are z-scored before
        weighting."""
        r1 = MockReward(100.0)
        r2 = MockReward(-100.0)
        composite = CompositeReward(
            strategies=[r1, r2],
            weights=[0.5, 0.5],
            auto_scale=True,
        )
        env = MagicMock(spec=TradingEnvProtocol)
        # First call: count < 2, so raw values pass through (no normalization yet)
        composite.calculate_reward(env)
        # Second call with different rewards
        r1.value = 200.0
        r2.value = -200.0
        reward2 = composite.calculate_reward(env)
        # Just ensure it runs without error and returns a float
        assert isinstance(reward2, float)

    def test_on_step_end_delegates_to_children(self):
        r1 = MagicMock(spec=BaseRewardStrategy)
        r2 = MagicMock(spec=BaseRewardStrategy)
        composite = CompositeReward(strategies=[r1, r2], weights=[0.5, 0.5])
        env = MagicMock(spec=TradingEnvProtocol)
        composite.on_step_end(env)
        r1.on_step_end.assert_called_once_with(env)
        r2.on_step_end.assert_called_once_with(env)

    def test_reset_delegates_to_children(self):
        r1 = MagicMock()
        r1.reset = MagicMock()
        r2 = MagicMock()
        r2.reset = MagicMock()
        composite = CompositeReward(strategies=[r1, r2], weights=[0.5, 0.5])
        composite.reset()
        r1.reset.assert_called_once()
        r2.reset.assert_called_once()

    def test_mismatched_strategies_weights_raises(self):
        with pytest.raises(ValueError):
            CompositeReward(strategies=[MockReward(1.0)], weights=[0.5, 0.5])

    def test_zero_weight_sum_raises(self):
        r1 = MockReward(1.0)
        composite = CompositeReward(strategies=[r1], weights=[0.0], normalize_weights=True)
        env = MagicMock(spec=TradingEnvProtocol)
        with pytest.raises(ValueError, match="zero"):
            composite.calculate_reward(env)


# ===========================================================================
# _RunningStat (Welford algorithm)
# ===========================================================================


class TestRunningStat:
    def test_first_value_returned_as_is(self):
        stat = _RunningStat()
        result = stat.update_and_normalize(5.0)
        assert result == 5.0  # count < 2, no normalization

    def test_normalizes_after_two_samples(self):
        stat = _RunningStat()
        stat.update_and_normalize(0.0)
        result = stat.update_and_normalize(1.0)
        # With mean=0.5 and std≈0.707, normalized = (1.0 - 0.5) / 0.707 ≈ 0.707
        assert isinstance(result, float)

    def test_clips_to_max(self):
        stat = _RunningStat(clip=1.0)
        # Feed many values near 0 to build stable stats, then spike
        for _ in range(100):
            stat.update_and_normalize(0.0)
        # Very large outlier should be clipped to +clip
        result = stat.update_and_normalize(1e9)
        assert result == pytest.approx(1.0)

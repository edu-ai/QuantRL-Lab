from unittest.mock import MagicMock

import pytest

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy, TradingEnvProtocol
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty


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

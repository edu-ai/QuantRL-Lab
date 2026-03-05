import gymnasium as gym
import numpy as np

from quantrl_lab.environments.core.interfaces import TradingEnvProtocol
from quantrl_lab.environments.stock.strategies.actions.standard import (
    StandardActionStrategy,
)


def test_action_space_creation():
    """Test that action space is created correctly."""
    strategy = StandardActionStrategy()
    action_space = strategy.define_action_space()

    assert isinstance(action_space, gym.spaces.Box)
    assert action_space.shape == (3,)
    assert action_space.dtype == np.float32


def test_action_handling(standard_env: TradingEnvProtocol):
    """Test that actions are handled correctly."""
    strategy = StandardActionStrategy()
    action = np.array([0.0, 0.5, 1.0], dtype=np.float32)  # Buy with 50% of balance at current price

    action_type, info = strategy.handle_action(standard_env, action)
    assert action_type is not None
    assert isinstance(info, dict)

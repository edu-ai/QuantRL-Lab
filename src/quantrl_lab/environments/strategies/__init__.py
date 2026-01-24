"""
Shared strategy interfaces for all trading environments.

Base classes for action, observation, and reward strategies that can be
reused across different asset types (stock, crypto, fx).
"""

from quantrl_lab.environments.strategies.actions import BaseActionStrategy
from quantrl_lab.environments.strategies.observations import BaseObservationStrategy
from quantrl_lab.environments.strategies.rewards import BaseRewardStrategy

__all__ = [
    "BaseActionStrategy",
    "BaseObservationStrategy",
    "BaseRewardStrategy",
]

from typing import Any, Dict, List, Type

from pydantic import BaseModel

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy
from quantrl_lab.environments.stock.strategies.rewards import CompositeReward


class RewardStrategyConfig(BaseModel):
    name: str
    strategy: Type[BaseRewardStrategy]
    params: Dict[str, Any] = {}

    def create_instance(self) -> BaseRewardStrategy:
        return self.strategy(**self.params)


class RewardCombination(BaseModel):
    name: str
    strategies: List[RewardStrategyConfig]
    weights: List[float]


def create_reward_strategy_from_combination(combination: "RewardCombination") -> CompositeReward:
    """
    Creates a CompositeReward instance from a RewardCombination.

    Args:
        combination (RewardCombination): The reward combination configuration.

    Returns:
        CompositeReward: The created weighted composite reward instance.
    """
    strategy_instances = [config.create_instance() for config in combination.strategies]
    return CompositeReward(strategies=strategy_instances, weights=combination.weights)

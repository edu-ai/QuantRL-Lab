from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np

# ==========================================
# Environment Protocol
# ==========================================


class TradingEnvProtocol(Protocol):
    """Protocol defining the interface for trading environments."""

    # Compulsory attributes for trading environments
    data: np.ndarray
    current_step: int
    price_column_index: int
    window_size: int
    action_space: gym.Space
    observation_space: gym.Space

    # Compulsory methods for trading environments
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]: ...
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]: ...
    def render(self, mode: str = "human"): ...
    def close(self): ...


# ==========================================
# Strategy Interfaces
# ==========================================


class BaseActionStrategy(ABC):
    """An abstract base class for defining action spaces and handling
    agent actions."""

    @abstractmethod
    def define_action_space(self) -> gym.spaces.Space:
        """
        Defines the action space for the environment.

        Returns:
            gym.spaces.Space: The action space for the environment.
        """
        pass

    @abstractmethod
    def handle_action(self, env_self: TradingEnvProtocol, action: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Handles the action taken by the agent in the environment.

        Args:
            env_self (TradingEnvProtocol): The environment instance where the action is taken.
            action (Any): The action taken by the agent.

        Returns:
            Tuple[Any, Dict[str, Any]]: The outcome of the action taken in the environment
        """
        pass


class BaseObservationStrategy(ABC):
    """Abstract base class for defining how an agent perceives the
    environment."""

    @abstractmethod
    def define_observation_space(self, env: TradingEnvProtocol) -> gym.spaces.Space:
        """
        Defines and returns the observation space for the environment.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            gym.spaces.Space: The observation space.
        """
        pass

    @abstractmethod
    def build_observation(self, env: TradingEnvProtocol) -> np.ndarray:
        """
        Builds the observation vector for the current state.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            np.ndarray: The observation vector.
        """
        pass

    @abstractmethod
    def get_feature_names(self, env: TradingEnvProtocol) -> List[str]:
        """
        Returns a list of feature names corresponding to the exact order
        of elements in the flattened observation vector.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            List[str]: A list of feature names (e.g., ["Close_t-1", "RSI_t", ...])
        """
        pass


class BaseRewardStrategy(ABC):
    """Abstract base class for calculating rewards."""

    @abstractmethod
    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the reward based on the action taken in the
        environment.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The calculated reward.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def on_step_end(self, env: TradingEnvProtocol):
        """Optional: A hook to update any internal state if needed."""
        pass

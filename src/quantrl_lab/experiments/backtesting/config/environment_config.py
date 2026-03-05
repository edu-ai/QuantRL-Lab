from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym


@dataclass
class BacktestEnvironmentConfig:
    """
    Configuration container for backtesting environments.

    This class enforces type safety for environment factories and allows
    attaching metadata (parameters) for experiment reproducibility.

    Attributes:
        train_env_factory: Callable returning the training environment
        test_env_factory: Callable returning the testing environment
        eval_env_factory: Optional callable returning an evaluation environment
        name: Identifier for this environment configuration (e.g., "BullMarket_HighVol")
        description: Human-readable description
        parameters: Dictionary of parameters used to create the environments
                    (window_size, strategies, etc.) - useful for logging/tracking
    """

    train_env_factory: Union[Callable[[], gym.Env], List[Callable[[], gym.Env]]]
    test_env_factory: Union[Callable[[], gym.Env], List[Callable[[], gym.Env]]]
    eval_env_factory: Optional[Union[Callable[[], gym.Env], List[Callable[[], gym.Env]]]] = None

    name: str = "default_env"
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate factories."""

        def _validate(factory, name):
            if factory is None:
                return
            if isinstance(factory, list):
                if not all(callable(f) for f in factory):
                    raise TypeError(f"All items in {name} list must be callable")
            elif not callable(factory):
                raise TypeError(f"{name} must be a callable or list of callables, got {type(factory)}")

        _validate(self.train_env_factory, "train_env_factory")
        _validate(self.test_env_factory, "test_env_factory")
        _validate(self.eval_env_factory, "eval_env_factory")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BacktestEnvironmentConfig":
        """
        Create configuration from a legacy dictionary.

        Args:
            config_dict: Dictionary with keys 'train_env_factory', 'test_env_factory', etc.

        Returns:
            BacktestEnvironmentConfig: The typed configuration object.
        """
        # Extract known keys
        train_factory = config_dict.get("train_env_factory")
        test_factory = config_dict.get("test_env_factory")
        eval_factory = config_dict.get("eval_env_factory")

        if not train_factory or not test_factory:
            raise ValueError("Dictionary must contain 'train_env_factory' and 'test_env_factory'")

        # Extract extra metadata if present
        name = config_dict.get("name", "legacy_dict_env")
        description = config_dict.get("description")
        parameters = config_dict.get("parameters", {})

        return cls(
            train_env_factory=train_factory,
            test_env_factory=test_factory,
            eval_env_factory=eval_factory,
            name=name,
            description=description,
            parameters=parameters,
        )

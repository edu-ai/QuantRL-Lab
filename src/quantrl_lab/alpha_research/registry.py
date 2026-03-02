from typing import Callable, Dict, List, Type

from .base import VectorizedTradingStrategy


class VectorizedStrategyRegistry:
    """Registry for vectorized trading strategies."""

    _strategies: Dict[str, Type[VectorizedTradingStrategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a strategy class."""

        def decorator(strategy_class: Type[VectorizedTradingStrategy]):
            cls._strategies[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> VectorizedTradingStrategy:
        """Create a strategy instance by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name](**kwargs)

    @classmethod
    def get(cls, name: str) -> Type[VectorizedTradingStrategy]:
        """Get the strategy class by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategies."""
        return list(cls._strategies.keys())

"""
Trading environments for reinforcement learning.

This package provides Gymnasium-compatible trading environments with
pluggable strategy patterns for actions, observations, and rewards.
"""

from quantrl_lab.environments import base, stock, strategies

from .exceptions import (
    ConfigurationError,
    DataFeedError,
    EnvironmentError,
    InvalidActionError,
    InvalidStateError,
)

__all__ = [
    "base",
    "stock",
    "strategies",
    "EnvironmentError",
    "InvalidActionError",
    "InvalidStateError",
    "ConfigurationError",
    "DataFeedError",
]

"""
QuantRL-Lab: A modular reinforcement learning framework for quantitative trading.

Main modules:
- environments: Trading environments with pluggable strategies
- data: Data acquisition and feature engineering
- experiments: Backtesting, feature engineering, hyperparameter tuning, and hedge pair screening
- deployment: Live trading
- utils: Shared utilities
"""

__version__ = "0.1.0"

from quantrl_lab import data, deployment, environments, experiments, utils

__all__ = ["data", "deployment", "environments", "experiments", "utils"]

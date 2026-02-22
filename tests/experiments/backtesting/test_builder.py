from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantrl_lab.alpha_research.models import AlphaResult
from quantrl_lab.environments.core.interfaces import (
    BaseActionStrategy,
    BaseObservationStrategy,
    BaseRewardStrategy,
)
from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder


@pytest.fixture
def mock_strategies():
    action = MagicMock(spec=BaseActionStrategy)
    reward = MagicMock(spec=BaseRewardStrategy)
    observation = MagicMock(spec=BaseObservationStrategy)
    return action, reward, observation


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame(
        {"Open": range(100), "High": range(100), "Low": range(100), "Close": range(100), "Volume": range(100)},
        index=dates,
    )
    return df


class TestBacktestEnvironmentBuilder:
    def test_initialization(self):
        builder = BacktestEnvironmentBuilder()
        assert builder._env_params["initial_balance"] == 100000.0

    def test_with_data(self, sample_data):
        builder = BacktestEnvironmentBuilder()
        builder.with_data(sample_data, sample_data)
        assert builder._train_data is not None
        assert builder._test_data is not None

    def test_with_strategies(self, mock_strategies):
        action, reward, observation = mock_strategies
        builder = BacktestEnvironmentBuilder()
        builder.with_strategies(action, reward, observation)
        assert builder._action_strategy == action

    def test_build_missing_components(self, sample_data, mock_strategies):
        action, reward, observation = mock_strategies
        builder = BacktestEnvironmentBuilder()

        # Missing data
        with pytest.raises(ValueError, match="Training and testing data must be provided"):
            builder.with_strategies(action, reward, observation).build()

        # Missing strategies
        builder = BacktestEnvironmentBuilder()
        with pytest.raises(ValueError, match="All strategies.*must be provided"):
            builder.with_data(sample_data, sample_data).build()

    def test_build_success(self, sample_data, mock_strategies):
        action, reward, observation = mock_strategies
        builder = BacktestEnvironmentBuilder()

        config = (
            builder.with_data(sample_data, sample_data)
            .with_strategies(action, reward, observation)
            .with_env_params(window_size=10)
            .build()
        )

        assert config.name == "default_env"  # Default name
        assert config.parameters["window_size"] == 10
        assert callable(config.train_env_factory)
        assert callable(config.test_env_factory)

        # Test factory execution
        env = config.train_env_factory()
        assert env is not None
        # Clean up
        env.close()

    @patch("quantrl_lab.experiments.backtesting.builder.results_to_pipeline_config")
    def test_with_alpha_signals(self, mock_bridge, sample_data, mock_strategies):
        action, reward, observation = mock_strategies

        # Mock alpha results
        alpha_res = MagicMock(spec=AlphaResult)
        mock_bridge.return_value = [{"SMA": {"window": 20}}]  # Mock config

        builder = BacktestEnvironmentBuilder()
        builder.with_data(sample_data, sample_data)
        builder.with_strategies(action, reward, observation)
        builder.with_alpha_signals([alpha_res])

        config = builder.build()

        assert config.parameters["alpha_signals_count"] == 1

        env = config.train_env_factory()

        # env.data is numpy array, check original_columns
        assert hasattr(env, "original_columns")
        assert "SMA_20" in env.original_columns or any("SMA" in c for c in env.original_columns)

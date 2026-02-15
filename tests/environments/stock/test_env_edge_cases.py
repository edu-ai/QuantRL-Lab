import numpy as np
import pandas as pd
import pytest

from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig
from quantrl_lab.environments.stock.single import SingleStockTradingEnv
from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward


@pytest.fixture
def base_config():
    return SingleStockEnvConfig(initial_balance=10000.0, window_size=10, max_episode_steps=50)


def test_dataframe_initialization(base_config):
    # Create simple DF with 'Close'
    df = pd.DataFrame({"Open": np.random.rand(100), "Close": np.random.rand(100)})

    env = SingleStockTradingEnv(
        data=df,
        config=base_config,
        action_strategy=StandardActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )

    # Should detect 'Close' automatically
    assert env.price_column_index == 1
    assert env.num_features == 2


def test_numpy_initialization_requires_column_index(base_config):
    data = np.random.rand(100, 2)

    # Ensure config doesn't provide a fallback
    base_config.price_column_index = None

    # Missing price_column -> Should fail
    with pytest.raises(ValueError):
        SingleStockTradingEnv(
            data=data,
            config=base_config,
            action_strategy=StandardActionStrategy(),
            reward_strategy=PortfolioValueChangeReward(),
            observation_strategy=FeatureAwareObservationStrategy(),
        )

    # Providing index -> Should work
    env = SingleStockTradingEnv(
        data=data,
        config=base_config,
        price_column=1,  # Index provided
        action_strategy=StandardActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )
    assert env.price_column_index == 1


def test_termination_logic(base_config):
    # Create tiny dataset (Window=10, Total=15)
    # Indices: 0..14.
    # Start step: 10.
    # Steps: 10, 11, 12, 13. (4 steps)
    # At end of step 13, current_step becomes 14, causing termination.
    df = pd.DataFrame({"Close": np.random.rand(15)})

    # Ensure config doesn't trigger truncation
    base_config.max_episode_steps = 50

    env = SingleStockTradingEnv(
        data=df,
        config=base_config,
        action_strategy=StandardActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )

    env.reset()

    # Step 1 (Index 10)
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert not terminated

    # Step 2 (Index 11)
    env.step(env.action_space.sample())

    # Step 3 (Index 12)
    env.step(env.action_space.sample())

    # Step 4 (Index 13) -> Should terminate
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())

    assert terminated
    assert not truncated


def test_truncation_logic(base_config):
    # Set tiny max_steps
    base_config.max_episode_steps = 3
    df = pd.DataFrame({"Close": np.random.rand(100)})

    env = SingleStockTradingEnv(
        data=df,
        config=base_config,
        action_strategy=StandardActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )

    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())

    # 3rd step -> Should truncate
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())

    assert not terminated
    assert truncated

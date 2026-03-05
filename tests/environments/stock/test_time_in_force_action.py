import gymnasium as gym
import numpy as np

from quantrl_lab.environments.stock import SingleStockTradingEnv
from quantrl_lab.environments.stock.components.portfolio import OrderTIF, OrderType
from quantrl_lab.environments.stock.strategies.actions.time_in_force import (
    TimeInForceActionStrategy,
)
from quantrl_lab.environments.stock.strategies.observations import (
    FeatureAwareObservationStrategy,
)
from quantrl_lab.environments.stock.strategies.rewards import PortfolioValueChangeReward


def test_tif_action_space_creation():
    """Test that TIF action space is created correctly."""
    strategy = TimeInForceActionStrategy()
    action_space = strategy.define_action_space()

    assert isinstance(action_space, gym.spaces.Box)
    assert action_space.shape == (4,)  # [type, amount, price, tif]
    assert action_space.dtype == np.float32


def test_handle_action_gtc(sample_data, stock_env_config):
    """Test that GTC orders are handled correctly."""
    data, _ = sample_data

    # Initialize env with TimeInForceActionStrategy
    env = SingleStockTradingEnv(
        data=data,
        config=stock_env_config,
        action_strategy=TimeInForceActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )

    # Reset to initialize portfolio
    env.reset()

    # Construct Action: Limit Buy (3/6 * 2 - 1 = 0.0), 10% amount, 0.9 price (below market), TIF=0 (GTC=-1.0)
    # Action type is normalized from [-1, 1] to [0, 6]: LimitBuy=3 -> (3/6)*2-1 = 0.0
    # TIF is normalized from [-1, 1] to [0, 2]: GTC=0 -> (0/2)*2-1 = -1.0
    action = np.array([0.0, 0.1, 0.9, -1.0], dtype=np.float32)

    env.step(action)

    # Verify order was placed
    assert len(env.portfolio.pending_orders) == 1
    order = env.portfolio.pending_orders[0]
    assert order.type == OrderType.LIMIT_BUY
    assert order.tif == OrderTIF.GTC


def test_handle_action_ioc(sample_data, stock_env_config):
    """Test that IOC orders are handled correctly."""
    data, _ = sample_data
    env = SingleStockTradingEnv(
        data=data,
        config=stock_env_config,
        action_strategy=TimeInForceActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )
    env.reset()

    # Action: Limit Buy (3/6 * 2 - 1 = 0.0), 10% amount, 0.5 price (WAY below market), TIF=1 (IOC=0.0)
    # IOC orders that don't fill immediately are cancelled — not added to pending_orders
    # Action type: LimitBuy=3 -> (3/6)*2-1 = 0.0
    # TIF: IOC=1 -> (1/2)*2-1 = 0.0
    action = np.array([0.0, 0.1, 0.5, 0.0], dtype=np.float32)

    env.step(action)

    # IOC orders that don't fill are cancelled immediately, so no pending orders
    assert len(env.portfolio.pending_orders) == 0


def test_handle_action_ttl(sample_data, stock_env_config):
    """Test that TTL orders are handled correctly."""
    data, _ = sample_data
    env = SingleStockTradingEnv(
        data=data,
        config=stock_env_config,
        action_strategy=TimeInForceActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    )
    env.reset()

    # Action: Limit Buy (3/6 * 2 - 1 = 0.0), 10% amount, 0.9 price, TIF=2 (TTL=1.0)
    # Action type: LimitBuy=3 -> (3/6)*2-1 = 0.0
    # TIF: TTL=2 -> (2/2)*2-1 = 1.0
    action = np.array([0.0, 0.1, 0.9, 1.0], dtype=np.float32)

    env.step(action)

    assert len(env.portfolio.pending_orders) == 1
    order = env.portfolio.pending_orders[0]
    assert order.tif == OrderTIF.TTL

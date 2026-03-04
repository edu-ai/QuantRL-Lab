# Custom Strategies

All three strategy types share the same pattern: inherit from a base class, implement the required methods, and inject into the environment. Base classes live in `quantrl_lab.environments.core.interfaces`.

## Custom Reward Strategy

```python
from quantrl_lab.environments.core.interfaces import BaseRewardStrategy, TradingEnvProtocol

class MyReward(BaseRewardStrategy):
    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """Return a scalar reward for the current step."""
        # env exposes: env.portfolio, env.current_step, env.data,
        #              env.price_column_index, env.window_size
        current_price = env._get_current_price()
        portfolio_value = env.portfolio.get_value(current_price)
        prev_value = getattr(self, "_prev_value", portfolio_value)
        reward = (portfolio_value - prev_value) / (prev_value + 1e-9)
        self._prev_value = portfolio_value
        return float(reward)

    def on_step_end(self, env: TradingEnvProtocol) -> None:
        """Optional hook called after each step (e.g. update internal state)."""
        pass

    def reset(self) -> None:
        """Called at the start of each episode."""
        self._prev_value = None
```

Inject it:
```python
env = SingleStockTradingEnv(..., reward_strategy=MyReward())
```

### Composing Rewards

Use `CompositeReward` to blend multiple signals without writing a new class:

```python
from quantrl_lab.environments.stock.strategies.rewards import (
    CompositeReward, DifferentialSortinoReward, TurnoverPenaltyReward, InvalidActionPenalty
)

reward_strategy = CompositeReward(
    strategies=[
        DifferentialSortinoReward(),
        TurnoverPenaltyReward(penalty_factor=0.001),
        InvalidActionPenalty(),
    ],
    weights=[1.0, 0.5, 0.2],
    normalize_weights=True,  # Automatically normalizes weights to sum to 1
    auto_scale=True,         # Normalizes each component to N(0,1) — recommended
)
```

## Custom Observation Strategy

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from quantrl_lab.environments.core.interfaces import BaseObservationStrategy, TradingEnvProtocol

class MyObservation(BaseObservationStrategy):
    NUM_PORTFOLIO_FEATURES = 2

    def define_observation_space(self, env: TradingEnvProtocol) -> gym.spaces.Space:
        n_features = env.window_size * env.num_features + self.NUM_PORTFOLIO_FEATURES
        return spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)

    def build_observation(self, env: TradingEnvProtocol) -> np.ndarray:
        # Market window
        start = max(0, env.current_step - env.window_size + 1)
        window = env.data[start : env.current_step + 1, :]

        # Pad if at episode start
        if len(window) < env.window_size:
            pad = np.repeat(window[:1], env.window_size - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)

        current_price = env._get_current_price()
        portfolio_features = np.array([
            env.portfolio.balance / env.portfolio.initial_balance,
            env.portfolio.total_shares * current_price / (env.portfolio.get_value(current_price) + 1e-9),
        ], dtype=np.float32)

        return np.concatenate([window.flatten(), portfolio_features])

    def get_feature_names(self, env: TradingEnvProtocol) -> list[str]:
        cols = getattr(env, "original_columns", [f"f{i}" for i in range(env.num_features)])
        names = [f"{c}_t-{env.window_size - 1 - i}" for i in range(env.window_size) for c in cols]
        names += ["balance_ratio", "position_ratio"]
        return names
```

## Custom Action Strategy

```python
import numpy as np
import gymnasium as gym
from quantrl_lab.environments.core.interfaces import BaseActionStrategy, TradingEnvProtocol
from quantrl_lab.environments.core.types import Actions

class MyActionStrategy(BaseActionStrategy):
    def define_action_space(self) -> gym.spaces.Space:
        # Discrete: 0=Hold, 1=Buy 10%, 2=Sell 10%
        return gym.spaces.Discrete(3)

    def handle_action(self, env_self: TradingEnvProtocol, action: int):
        action_map = {0: Actions.Hold, 1: Actions.Buy, 2: Actions.Sell}
        action_type = action_map.get(int(action), Actions.Hold)
        current_price = env_self._get_current_price()

        if action_type == Actions.Buy:
            env_self.portfolio.execute_market_order(action_type, current_price, 0.1, env_self.current_step)
        elif action_type == Actions.Sell and env_self.portfolio.total_shares > 0:
            env_self.portfolio.execute_market_order(action_type, current_price, 1.0, env_self.current_step)

        return action_type, {"type": action_type.name}
```

## Built-in Strategies Reference

### Action Strategies

| Class | Module | Action Space | Description |
|-------|--------|-------------|-------------|
| `StandardActionStrategy` | `strategies.actions.standard` | Box(3,) continuous | Full action space with limit/stop orders |
| `TimeInForceActionStrategy` | `strategies.actions.time_in_force` | Box(3,) continuous | Adds time-in-force semantics |

### Observation Strategies

| Class | Module | Description |
|-------|--------|-------------|
| `FeatureAwareObservationStrategy` | `strategies.observations.feature_aware` | Smart normalization by feature type; 9 portfolio features |

### Reward Strategies

| Class | Module | Description |
|-------|--------|-------------|
| `PortfolioValueChangeReward` | `strategies.rewards.portfolio_value` | Raw portfolio value change |
| `DifferentialSortinoReward` | `strategies.rewards.sortino` | Risk-adjusted (downside volatility) |
| `DifferentialSharpeReward` | `strategies.rewards.sharpe` | Risk-adjusted (total volatility) |
| `DrawdownPenaltyReward` | `strategies.rewards.drawdown` | Penalizes max drawdown |
| `TurnoverPenaltyReward` | `strategies.rewards.turnover` | Penalizes excessive trading |
| `InvalidActionPenalty` | `strategies.rewards.invalid_action` | Penalizes impossible actions |
| `BoredomPenaltyReward` | `strategies.rewards.boredom` | Penalizes prolonged inactivity |
| `LimitExecutionReward` | `strategies.rewards.execution_bonus` | Rewards price improvement achieved by limit orders vs market price |
| `OrderExpirationPenaltyReward` | `strategies.rewards.expiration` | Penalizes expired orders |
| `CompositeReward` | `strategies.rewards.composite` | Weighted combination of the above |

## See Also

- [Reward Shaping](reward_shaping.md) — guidance on designing reward functions
- [Backtesting](backtesting.md) — running experiments with custom strategies

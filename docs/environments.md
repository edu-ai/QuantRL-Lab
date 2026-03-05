# Trading Environments

QuantRL-Lab wraps financial markets as [Gymnasium](https://gymnasium.farama.org/) environments. The agent receives an observation of the market and its portfolio state, chooses an action, and receives a scalar reward — the standard RL loop.

---

## Available Environments

| Environment | Status | Description |
|---|---|---|
| `SingleStockTradingEnv` | **Stable** | Trades a single asset. Full order-type support. |
| `MultiStockTradingEnv` | Planned | Portfolio of multiple assets with cross-asset observations. |
| Futures / Options envs | Roadmap | Contract expiry, margin, and derivatives semantics. |

---

## `SingleStockTradingEnv`

The core environment. It delegates all algorithmic decisions to three injected strategy objects, keeping the environment itself thin and stable.

### Constructor

```python
SingleStockTradingEnv(
    data: pd.DataFrame | np.ndarray,
    config: SingleStockEnvConfig,
    action_strategy: BaseActionStrategy,
    reward_strategy: BaseRewardStrategy,
    observation_strategy: BaseObservationStrategy,
    price_column: str | int | None = None,   # auto-detected if None
)
```

### Configuration

See [Configuration](getting-started/configuration.md) for the full parameter reference. The key fields:

```python
from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig, SimulationConfig

config = SingleStockEnvConfig(
    initial_balance=100_000,
    window_size=20,
    simulation=SimulationConfig(transaction_cost_pct=0.001, slippage=0.001),
)
```

### Minimal working example

```python
import pandas as pd
from stable_baselines3 import PPO

from quantrl_lab.data.sources import YFinanceDataLoader
from quantrl_lab.data.processing.processor import DataProcessor
from quantrl_lab.environments.stock.single import SingleStockTradingEnv
from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig
from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward

# 1. Get data
loader = YFinanceDataLoader()
raw_df = loader.get_historical_ohlcv_data(["AAPL"], start="2021-01-01", end="2024-01-01")

processor = DataProcessor(ohlcv_data=raw_df)
df, _ = processor.data_processing_pipeline(indicators=["SMA", "RSI", "MACD"])

# 2. Build environment
config = SingleStockEnvConfig(initial_balance=100_000, window_size=20)

env = SingleStockTradingEnv(
    data=df,
    config=config,
    action_strategy=StandardActionStrategy(),
    reward_strategy=PortfolioValueChangeReward(),
    observation_strategy=FeatureAwareObservationStrategy(),
)

# 3. Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)
```

---

## Action Space

The action space is defined by the action strategy. Two strategies are available.

### `StandardActionStrategy` — 3-element Box

The default strategy. Produces a **`Box(3,)`** action:

```
action = [action_type, amount, price_modifier]
```

| Dimension | Range | Meaning |
|---|---|---|
| `action_type` | `[-1, 1]` → mapped to `[0, 6]` | Which order type to place |
| `amount` | `[-1, 1]` → mapped to `[0, 1]` | Fraction of available balance/shares |
| `price_modifier` | `[0.9, 1.1]` | Limit price as a multiple of market price |

The symmetric `[-1, 1]` range for `action_type` and `amount` is intentional — an untrained agent outputs values near 0, which maps to the middle of the action range rather than always choosing "Hold" or 0% amount.

**Action types (`Actions` enum):**

| Int | Name | Description |
|---|---|---|
| 0 | `Hold` | No action |
| 1 | `Buy` | Market buy `amount%` of available cash |
| 2 | `Sell` | Market sell `amount%` of held shares |
| 3 | `LimitBuy` | Place limit buy order at `price × price_modifier` |
| 4 | `LimitSell` | Place limit sell order at `price × price_modifier` |
| 5 | `StopLoss` | Place stop-loss order at `price × price_modifier` |
| 6 | `TakeProfit` | Place take-profit order at `price × price_modifier` |

```python
from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
import numpy as np

action_strategy = StandardActionStrategy()
print(action_strategy.define_action_space())
# Box([-1.  -1.   0.9], [1.  1.  1.1], (3,), float32)

# Manual action examples:
# Market buy 50% of cash:
action = np.array([-0.333, 0.0, 1.0])   # action_type → 2 (Buy), amount → 0.5

# Limit sell 75% of shares at 3% above market:
action = np.array([0.333, 0.5, 1.03])   # action_type → 4 (LimitSell), amount → 0.75

# Stop-loss 100% of shares at 5% below market:
action = np.array([0.667, 1.0, 0.95])   # action_type → 5 (StopLoss), amount → 1.0
```

---

### `TimeInForceActionStrategy` — 4-element Box

An extended strategy that adds explicit **Time-In-Force (TIF)** control as a 4th dimension. Useful when the agent needs to differentiate between persistent and short-lived limit orders.

```
action = [action_type, amount, price_modifier, tif_type]
```

| Dimension | Range | Meaning |
|---|---|---|
| `action_type` | `[-1, 1]` → `[0, 6]` | Same as Standard |
| `amount` | `[0, 1]` | Fraction of available balance/shares |
| `price_modifier` | `[0.9, 1.1]` | Limit price multiplier |
| `tif_type` | `[-1, 1]` → `{0, 1, 2}` | Order lifetime policy |

**TIF types:**

| Int | Name | Behaviour |
|---|---|---|
| 0 | `GTC` (Good Till Cancelled) | Order persists until filled or cancelled |
| 1 | `IOC` (Immediate or Cancel) | Fills immediately or is cancelled |
| 2 | `TTL` (Time To Live) | Expires after `order_expiration_steps` steps |

```python
from quantrl_lab.environments.stock.strategies.actions.time_in_force import TimeInForceActionStrategy

action_strategy = TimeInForceActionStrategy()
print(action_strategy.define_action_space())
# Box([-1.   0.   0.9 -1. ], [1.  1.  1.1  1. ], (4,), float32)
```

---

## Observation Space

The default observation strategy is `FeatureAwareObservationStrategy`. It produces a **flat `Box(N,)`** vector:

```
observation = [market_window (flattened), portfolio_features]
```

Total size: `window_size × num_features + 9`

### Market window

A rolling window of `window_size` steps (oldest → newest) of the full feature matrix. Columns are normalized by type:

| Column type | Normalisation | Examples |
|---|---|---|
| Price-like | Divided by first value in window (relative return scale) | `open`, `high`, `low`, `close`, `SMA`, `EMA`, `BB_upper` |
| Oscillators 0–100 | Divided by 100 | `RSI`, `STOCH`, `MFI`, `ADX` |
| Williams %R (−100–0) | `(x + 100) / 100` | `WILLR` |
| CCI (unbounded ~±200) | Divided by 200 | `CCI` |
| ATR | `ATR / close` (% volatility) | `ATR` |
| MACD | `MACD / close` (scale-free) | `MACD`, `MACD_signal` |
| OBV | Z-scored within the window | `OBV` |
| Sentiment / analyst / sector | Passed through as-is | `sentiment_score`, `sector_change` |

Detection is keyword-based on column names — no manual labelling needed.

### Portfolio features (always appended, 9 values)

| Feature | Description |
|---|---|
| `portfolio_balance_ratio` | `cash / initial_balance` |
| `position_size_ratio` | `(shares × price) / portfolio_value` |
| `unrealized_pl_pct` | `(current_price − avg_entry) / avg_entry` |
| `price_pos_in_range` | Position of current price within recent high/low range [0, 1] |
| `recent_volatility` | Std-dev of returns over `volatility_lookback` steps |
| `recent_trend` | Linear slope of price over `trend_lookback` steps |
| `risk_reward_ratio` | `(avg_take_profit − price) / (price − avg_stop_loss)` |
| `dist_to_stop_loss` | `(price − avg_stop_loss) / price` |
| `dist_to_take_profit` | `(avg_take_profit − price) / price` |

```python
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy

obs_strategy = FeatureAwareObservationStrategy(
    volatility_lookback=10,   # steps for recent volatility calc
    trend_lookback=10,        # steps for recent trend calc
    normalize_stationary=True # scale oscillators like RSI, ADX to [0,1]
)

# After env is built:
feature_names = obs_strategy.get_feature_names(env)
print(feature_names[:5])
# ['open_t-19', 'high_t-19', 'low_t-19', 'close_t-19', 'volume_t-19']
print(feature_names[-9:])
# ['portfolio_balance_ratio', 'position_size_ratio', ...]
```

---

## Reward Shaping

Rewards are pluggable. Every strategy inherits `BaseRewardStrategy` and implements `calculate_reward(env) -> float`. Strategies can also implement `on_step_end(env)` for stateful updates and `reset()` for between-episode cleanup.

### Available reward strategies

#### `PortfolioValueChangeReward`

The simplest baseline. Reward = % change in portfolio value since the previous step.

```python
from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward

reward_strategy = PortfolioValueChangeReward()
# reward = (current_value - prev_value) / prev_value
```

---

#### `DifferentialSharpeReward`

Dense step-level Sharpe signal. Scales the current excess return by the historical volatility, so large returns in a low-volatility regime score higher than the same return in a volatile one.

```python
from quantrl_lab.environments.stock.strategies.rewards.sharpe import DifferentialSharpeReward

reward_strategy = DifferentialSharpeReward(
    risk_free_rate=0.0,   # per-step risk-free rate (usually 0 for daily)
    decay=0.99,           # EMA decay for running mean/variance; closer to 1 = longer memory
)
# reward = excess_return / running_std_dev
```

---

#### `DifferentialSortinoReward`

Like Sharpe but penalises only **downside** volatility (returns below `target_return`). Upside volatility is not penalised, so the agent is not discouraged from large positive moves.

```python
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward

reward_strategy = DifferentialSortinoReward(
    target_return=0.0,   # minimum acceptable return (MAR)
    decay=0.99,
)
# reward = current_return / running_downside_deviation
```

---

#### `DrawdownPenaltyReward`

Continuous penalty proportional to the current drawdown from the episode's high-water mark. Keeps persistent pressure on the agent to recover from losses.

```python
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward

reward_strategy = DrawdownPenaltyReward(penalty_factor=1.0)
# reward = -(drawdown_pct * penalty_factor)
```

---

#### `TurnoverPenaltyReward`

Penalises excessive trading by amplifying the transaction costs already embedded in portfolio value. Discourages "churn" trades whose P&L is swamped by fees.

```python
from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

reward_strategy = TurnoverPenaltyReward(penalty_factor=2.0)
# reward = -(fees_paid_this_step * penalty_factor)
# penalty_factor=1.0 doubles the cost impact; 5.0 aggressively punishes churning
```

---

#### `InvalidActionPenalty`

Fixed penalty when the agent tries an illegal action — e.g. selling with no shares held. Teaches the agent feasibility faster than relying on P&L alone.

```python
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty

reward_strategy = InvalidActionPenalty(penalty=-0.5)
# reward = -0.5 if action was invalid, else 0.0
```

---

#### `BoredomPenaltyReward`

Penalises holding a position beyond a grace period without meaningful price movement. Encourages timely entries and exits rather than indefinite holding.

```python
from quantrl_lab.environments.stock.strategies.rewards.boredom import BoredomPenaltyReward

reward_strategy = BoredomPenaltyReward(
    penalty_per_step=-0.001,  # small penalty per step after grace period
    grace_period=10,          # steps before penalty kicks in
    min_profit_pct=0.005,     # unrealized profit % that would "reset" the timer
)
```

---

#### `LimitExecutionReward`

Bonus when a limit order fills at a better price than the prevailing market price at the time it was placed. Rewards the agent for using limit orders effectively rather than always paying market.

```python
from quantrl_lab.environments.stock.strategies.rewards.execution_bonus import LimitExecutionReward

reward_strategy = LimitExecutionReward(improvement_multiplier=10.0)
# reward = price_improvement_pct * improvement_multiplier
# e.g. limit buy 2% below market → +0.20 bonus
```

---

#### `OrderExpirationPenaltyReward`

Fixed penalty per expired pending order. Discourages "order spam" — placing unrealistic limit orders that never fill and clog the system until TTL.

```python
from quantrl_lab.environments.stock.strategies.rewards.expiration import OrderExpirationPenaltyReward

reward_strategy = OrderExpirationPenaltyReward(penalty_per_order=-0.1)
# reward = num_expired_orders * penalty_per_order
```

---

### Combining with `CompositeReward`

Most real experiments combine multiple components. `CompositeReward` handles weighting and optional per-component normalisation.

```python
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty

reward_strategy = CompositeReward(
    strategies=[
        DifferentialSortinoReward(decay=0.99),
        DrawdownPenaltyReward(penalty_factor=0.5),
        TurnoverPenaltyReward(penalty_factor=1.0),
        InvalidActionPenalty(penalty=-0.5),
    ],
    weights=[0.6, 0.2, 0.1, 0.1],
    normalize_weights=True,   # weights are normalised to sum to 1 (default)
    auto_scale=False,         # if True, each component is z-scored before weighting
)
```

**`normalize_weights=True`** (default): weights are scaled so they always sum to 1 even if they don't already.

**`auto_scale=True`**: each component is standardised to N(0,1) via Welford's online algorithm before being weighted. Use this when components have very different natural scales and you can't hand-tune weights easily. Running stats persist across episodes for stability.

---

## Full End-to-End Example

A complete pipeline: fetch data → process → build env → train → evaluate.

```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from quantrl_lab.data.sources import YFinanceDataLoader
from quantrl_lab.data.processing.processor import DataProcessor
from quantrl_lab.environments.stock.single import SingleStockTradingEnv
from quantrl_lab.environments.stock.components.config import (
    SingleStockEnvConfig, SimulationConfig, RewardConfig
)
from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty

# ── 1. Data ──────────────────────────────────────────────────────────────────
loader = YFinanceDataLoader()
raw_df = loader.get_historical_ohlcv_data(["AAPL"], start="2019-01-01", end="2024-01-01")

processor = DataProcessor(ohlcv_data=raw_df)
splits, meta = processor.data_processing_pipeline(
    indicators=["RSI", {"SMA": {"window": 50}}, {"EMA": {"window": 20}}, "ATR", "MACD"],
    split_config={"train": 0.8, "test": 0.2},
)
train_df, test_df = splits["train"], splits["test"]

# ── 2. Strategies ─────────────────────────────────────────────────────────────
action_strategy  = StandardActionStrategy()
obs_strategy     = FeatureAwareObservationStrategy(volatility_lookback=10, trend_lookback=10)
reward_strategy  = CompositeReward(
    strategies=[
        DifferentialSortinoReward(decay=0.99),
        DrawdownPenaltyReward(penalty_factor=0.5),
        TurnoverPenaltyReward(penalty_factor=1.0),
        InvalidActionPenalty(penalty=-0.5),
    ],
    weights=[0.6, 0.2, 0.1, 0.1],
)

# ── 3. Config ─────────────────────────────────────────────────────────────────
config = SingleStockEnvConfig(
    initial_balance=100_000,
    window_size=20,
    simulation=SimulationConfig(transaction_cost_pct=0.001, slippage=0.001),
    rewards=RewardConfig(clip_range=(-1.0, 1.0)),
)

# ── 4. Train environment ───────────────────────────────────────────────────────
def make_train_env():
    return SingleStockTradingEnv(
        data=train_df, config=config,
        action_strategy=action_strategy,
        reward_strategy=reward_strategy,
        observation_strategy=obs_strategy,
    )

train_env = DummyVecEnv([make_train_env])
model = PPO("MlpPolicy", train_env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=200_000)

# ── 5. Evaluate on test set ───────────────────────────────────────────────────
test_env = SingleStockTradingEnv(
    data=test_df, config=config,
    action_strategy=StandardActionStrategy(),
    reward_strategy=DifferentialSortinoReward(),   # use clean strategy for eval
    observation_strategy=FeatureAwareObservationStrategy(),
)

obs, _ = test_env.reset()
portfolio_values = []

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    portfolio_values.append(info["portfolio_value"])
    if terminated or truncated:
        break

final_value = portfolio_values[-1]
total_return = (final_value - 100_000) / 100_000
print(f"Final portfolio value: ${final_value:,.2f}  ({total_return:.1%})")
```

---

## Custom Reward Strategy

Extend `BaseRewardStrategy` to implement any reward logic:

```python
from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

class CalmarRatioReward(BaseRewardStrategy):
    """Reward = annualised return / max drawdown (Calmar-inspired)."""

    def __init__(self, penalty_factor: float = 1.0):
        super().__init__()
        self.penalty_factor = penalty_factor
        self._peak_value = 0.0
        self._max_drawdown = 1e-9
        self._cumulative_return = 0.0
        self._step = 0

    def calculate_reward(self, env) -> float:
        price = env._get_current_price()
        value = env.portfolio.get_value(price)

        if self._peak_value < value:
            self._peak_value = value

        dd = (self._peak_value - value) / (self._peak_value + 1e-9)
        self._max_drawdown = max(self._max_drawdown, dd)

        ret = (value - env.prev_portfolio_value) / (env.prev_portfolio_value + 1e-9)
        self._cumulative_return += ret
        self._step += 1

        calmar = self._cumulative_return / (self._max_drawdown * self.penalty_factor)
        return float(calmar / (self._step + 1))   # normalise by episode length

    def reset(self):
        self._peak_value = 0.0
        self._max_drawdown = 1e-9
        self._cumulative_return = 0.0
        self._step = 0
```

---

## Roadmap

| Feature | Status | Notes |
|---|---|---|
| `MultiStockTradingEnv` | In development | Portfolio over N assets; cross-sectional observations |
| Continuous position sizing | Planned | Fractional shares, no rounding |
| Futures environment | Planned | Margin, leverage, contract expiry |
| Options environment | Exploratory | Greeks, IV surface as observations |
| Live trading integration | Partial | `AlpacaTrader` in `deployment/trading/` handles live execution |

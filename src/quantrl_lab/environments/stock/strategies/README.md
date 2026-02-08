# Stock Trading Strategies

This directory contains the pluggable components that define the behavior of the `SingleStockTradingEnv`. The environment is designed around the **Strategy Pattern**, allowing you to swap out how the agent acts, what it sees, and how it is rewarded without modifying the core environment code.

## Architecture

A complete trading agent configuration consists of three distinct strategies:

### 1. Actions (`strategies/actions/`)
Defines the **output** of the agent (the decision).

*   **`StandardActionStrategy`**: The default recommended strategy.
    *   **Continuous Space**: Outputs `[ActionType, Amount%, PriceModifier]`.
    *   **Capabilities**: Can execute Market orders, Limit orders, and Risk Management orders (Stop Loss/Take Profit).
    *   **Synergy**: Works best when the observation space provides volatility and regime signals to inform the `PriceModifier` (limit price) and `Amount%`.

### 2. Observations (`strategies/observations/`)
Defines the **input** to the agent (the state).

*   **`FeatureAwareObservationStrategy`** (Recommended):
    *   **Smart Normalization**: Distinguishes between **Price Data** (normalized relative to window start) and **Stationary Indicators** (RSI, Sentiment, Time Features - preserved as raw/absolute values).
    *   **Cyclical Time**: Incorporates `day_sin`, `day_cos` etc. to learn seasonality.
    *   **Portfolio State**: Includes critical context like Unrealized P/L, Risk/Reward ratios, and distances to stop/profit levels.

### 3. Rewards (`strategies/rewards/`)
Defines the **objective** of the agent (the optimization goal).

*   **`DifferentialSortinoReward`**: The primary engine.
    *   Optimizes for **Risk-Adjusted Returns**.
    *   Penalizes downside volatility (losses) while rewarding upside volatility (gains).
    *   Naturally encourages good position sizing and timely exits without rigid rules.
*   **`InvalidActionPenalty`**: A soft constraint to discourage technical errors (e.g., selling shares not owned).

## Recommended Configuration

To create a robust, production-grade RL environment, combine these strategies as follows:

```python
from quantrl_lab.environments.stock.strategies.actions import StandardActionStrategy
from quantrl_lab.environments.stock.strategies.observations import FeatureAwareObservationStrategy
from quantrl_lab.environments.stock.strategies.rewards import (
    WeightedCompositeReward,
    DifferentialSortinoReward,
    InvalidActionPenalty
)

# 1. Action: The standard continuous space
action_strategy = StandardActionStrategy()

# 2. Observation: The "Smart" view of the market
observation_strategy = FeatureAwareObservationStrategy(
    volatility_lookback=10,
    trend_lookback=10,
    normalize_stationary=True
)

# 3. Reward: The "Risk-Adjusted" Objective
# 90% focus on Sortino Ratio, 10% on avoiding invalid actions
reward_strategy = WeightedCompositeReward(
    strategies=[
        DifferentialSortinoReward(target_return=0.0, decay=0.99),
        InvalidActionPenalty(penalty=-0.1)
    ],
    weights=[0.9, 0.1]
)

# Create the environment
env_config = BacktestRunner.create_env_config_factory(
    # ... data params ...
    action_strategy=action_strategy,
    observation_strategy=observation_strategy,
    reward_strategy=reward_strategy
)
```

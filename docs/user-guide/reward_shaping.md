# Reward Shaping for Financial Agents

In Reinforcement Learning (RL), the **reward function** defines the goal. In finance, simply rewarding "profit" ($ P_t - P_{t-1} $) is often insufficient because financial returns are **noisy, sparse, and non-stationary**.

If your agent is losing money or failing to converge, "shaping" the reward function to provide denser, more stable feedback is often the solution.

## The Challenge with Raw Returns

1.  **Noise:** A trade might be profitable due to market luck, not skill. The agent struggles to distinguish the two.
2.  **Sparsity:** If an agent holds for 10 days and sells, the PnL appears only at step 10. The previous 9 steps have 0 reward (credit assignment problem).
3.  **Risk Ignorance:** Maximizing raw return often leads to reckless leveraging.

## Strategies for Improvement

### 1. Risk-Adjusted Rewards
Instead of raw PnL, reward the **quality** of the return relative to its volatility.

*   **Differential Sharpe Ratio:** Rewards returns that increase the portfolio's Sharpe Ratio.
*   **Differential Sortino Ratio:** Similar to Sharpe, but only penalizes *downside* volatility. This is often better for trading because we don't want to penalize "good" volatility (upside spikes).

*QuantRL-Lab provides `DifferentialSortinoReward` out of the box.*

### 2. Composite Rewards (Mixing Signals)
Rarely does a single metric capture everything. You often want a blend of objectives. Use the `CompositeReward` strategy to combine multiple signals.

**Example Mix:**
*   **Primary Goal:** Sortino Ratio (Weight: 1.0)
*   **Auxiliary Goal:** Forecast Accuracy (Weight: 0.5) - "Did the price move in the direction of my trade?"
*   **Penalty:** Turnover/Transaction Costs (Weight: -0.1) - "Don't overtrade."

```python
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

# Define the mix
reward_strategy = CompositeReward(
    strategies=[
        DifferentialSortinoReward(),
        TurnoverPenaltyReward(penalty_factor=0.001)
    ],
    weights=[1.0, 0.5],
    auto_scale=True  # Crucial: Normalizes components to N(0,1) so weights are meaningful
)
```

### 3. Auxiliary Tasks (Dense Rewards)
To help the agent learn faster, give it "hints" even if they aren't the final objective.

*   **Directional Accuracy:** Give a small positive reward (+0.1) if `sign(action) == sign(next_return)`, regardless of transaction costs. This helps the agent learn market dynamics before it learns regarding PnL.
*   **Holding Penalty:** A tiny negative reward for every step (e.g., -0.0001). This encourages the agent to make efficient use of capital rather than doing nothing (though use with caution, as it forces activity).

### 4. Reward Normalization
Neural networks learn best when targets are roughly in the range `[-1, 1]` or `[-5, 5]`.
*   **Raw PnL** might be +$1000 one day and -$500 the next. These large values destabilize gradients.
*   **Solution:** Enable `auto_scale=True` in `CompositeReward`. This uses a running mean/variance tracker (Welford's algorithm) to standardize rewards dynamically.

## Recommended "Recipe" for Stability

If your agent is unstable or losing money:

1.  **Start Simple:** Use `DifferentialSortinoReward`. It naturally handles risk.
2.  **Add Normalization:** Ensure observations and rewards are normalized.
3.  **Add a Penalty:** If the agent churns (buys/sells rapidly), add a `TurnoverPenaltyReward`.
4.  **Symmetric Actions:** Ensure your Action Strategy uses symmetric scaling (already implemented in `StandardActionStrategy` and `TimeInForceActionStrategy`) so the agent explores Buy/Sell equally.

## Code Example: Custom Reward Strategy

You can easily define a custom strategy by inheriting from `BaseRewardStrategy`.

```python
import numpy as np
from quantrl_lab.environments.core.interfaces import BaseRewardStrategy
from quantrl_lab.environments.core.types import Actions

class TrendFollowingReward(BaseRewardStrategy):
    def calculate_reward(self, env):
        # Reward buying when trend is up, selling when trend is down
        # regardless of PnL (Auxiliary task)

        # Compute a simple trend from recent close prices
        price_col = env.price_column_index
        start = max(0, env.current_step - 10)
        recent_prices = env.data[start : env.current_step + 1, price_col]

        trend = 0.0
        if len(recent_prices) >= 2:
            trend = float(np.polyfit(range(len(recent_prices)), recent_prices, 1)[0])

        action_type = env.action_type  # set by action_strategy.handle_action() each step
        if action_type == Actions.Buy and trend > 0:
            return 0.1
        elif action_type == Actions.Sell and trend < 0:
            return 0.1
        return 0.0
```

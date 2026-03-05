# Reward Strategies

This directory contains strategies for calculating the reward signal, which guides the agent's learning process.

## Philosophy: Risk-Adjusted Returns

Modern financial RL should focus on **risk-adjusted returns** rather than raw profit or arbitrary rule-based constraints.

*   **Old Approach:** Combine "Profit Reward" + "Penalty for big trades" + "Penalty for low cash" + ...
    *   *Problem:* Hard to tune weights; agent gets confused by conflicting signals.
*   **New Approach:** Maximize **Differential Sortino Ratio** combined with specific behavior-shaping penalties.
    *   *Benefit:* The Sortino Ratio inherently rewards high returns while penalizing *downside* volatility (risk). The agent naturally learns to size positions correctly and cut losses. Specific penalties discourage "spamming" the system or churning the account.

## Available Strategies

### 1. `DifferentialSortinoReward` (Recommended Primary)
*   **Goal:** Maximize the Sortino Ratio at every step.
*   **Mechanism:** Calculates the incremental contribution of the current step's return to the running Sortino Ratio.
*   **Usage:** Use this as your main objective (weight ~0.9 or 1.0).

### 2. `InvalidActionPenalty` (Guard Rail)
*   **Goal:** Discourage technical errors.
*   **Mechanism:** Applies a fixed penalty (e.g., -0.1 or -1.0) if the agent tries to sell shares it doesn't own.
*   **Usage:** Combine with Sortino using `CompositeReward`.

### 3. `TurnoverPenaltyReward` (Efficiency)
*   **Goal:** Discourage excessive trading ("churning").
*   **Mechanism:** Applies a penalty proportional to the transaction fees paid in the current step (e.g., `fees * 5.0`).
*   **Usage:** Use to force the agent to find high-conviction trades that justify the cost.

### 4. `OrderExpirationPenaltyReward` (System Health)
*   **Goal:** Discourage placing unrealistic limit orders ("order spam").
*   **Mechanism:** Applies a small penalty for every order that expires without execution.
*   **Usage:** Keeps the agent focused on realistic pricing near the market.

### 5. `PortfolioValueChangeReward` (Legacy/Basic)
*   **Goal:** Maximize raw profit.
*   **Mechanism:** Reward = % change in portfolio value.
*   **Usage:** Use for simple baseline tests or curriculum learning (Stage 1), but lacks risk awareness.

### 6. `CompositeReward`
*   **Goal:** Combine multiple signals.
*   **Features:** Supports `auto_scale=True` to normalize component rewards to N(0,1) dynamically.
*   **Usage:**
    ```python
    reward_strategy = CompositeReward(
        strategies=[
            DifferentialSortinoReward(),   # Main Driver
            TurnoverPenaltyReward(5.0),    # Efficiency Check
            OrderExpirationPenaltyReward() # Sanitation
        ],
        weights=[0.8, 0.1, 0.1],
        auto_scale=True
    )
    ```

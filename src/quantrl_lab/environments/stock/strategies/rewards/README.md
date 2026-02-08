# Reward Strategies

This directory contains strategies for calculating the reward signal, which guides the agent's learning process.

## Philosophy: Risk-Adjusted Returns

Modern financial RL should focus on **risk-adjusted returns** rather than raw profit or arbitrary rule-based constraints.

*   **Old Approach:** Combine "Profit Reward" + "Penalty for big trades" + "Penalty for low cash" + ...
    *   *Problem:* Hard to tune weights; agent gets confused by conflicting signals.
*   **New Approach:** Maximize **Differential Sortino Ratio**.
    *   *Benefit:* The Sortino Ratio inherently rewards high returns while penalizing *downside* volatility (risk). The agent naturally learns to size positions correctly and cut losses to improve this metric, without needing explicit rules.

## Available Strategies

### 1. `DifferentialSortinoReward` (Recommended Primary)
*   **Goal:** Maximize the Sortino Ratio at every step.
*   **Mechanism:** Calculates the incremental contribution of the current step's return to the running Sortino Ratio.
*   **Usage:** Use this as your main objective (weight ~0.9 or 1.0).

### 2. `InvalidActionPenalty` (Guard Rail)
*   **Goal:** Discourage technical errors.
*   **Mechanism:** Applies a fixed penalty (e.g., -0.1 or -1.0) if the agent tries to sell shares it doesn't own.
*   **Usage:** Combine with Sortino using `WeightedCompositeReward`.

### 3. `PortfolioValueChangeReward` (Legacy/Basic)
*   **Goal:** Maximize raw profit.
*   **Mechanism:** Reward = % change in portfolio value.
*   **Usage:** Use for simple baseline tests or curriculum learning (Stage 1), but lacks risk awareness.

### 4. `WeightedCompositeReward`
*   **Goal:** Combine multiple signals.
*   **Usage:**
    ```python
    reward_strategy = WeightedCompositeReward(
        strategies=[
            DifferentialSortinoReward(),   # Main Driver
            InvalidActionPenalty()         # Safety Check
        ],
        weights=[0.9, 0.1]
    )
    ```

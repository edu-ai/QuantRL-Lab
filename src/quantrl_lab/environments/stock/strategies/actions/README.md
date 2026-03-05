# Action Strategies & Symmetric Scaling

This directory contains the action strategies for the stock trading environment. These strategies translate the raw, continuous action vector from the RL agent into concrete trading instructions for the portfolio.

## Critical Implementation Detail: Symmetric Scaling

To ensure effective exploration, especially with policy gradient algorithms like PPO (Proximal Policy Optimization), we use **symmetric scaling** for categorical action dimensions.

### The Problem: Initialization Bias
Neural networks typically initialize with small weights, resulting in output values (action means) close to **0.0**.

If we used a naive mapping where `[0, N]` maps directly to action indices:
*   Action 0 (`Hold`) would require an output of ~0.
*   Action 1 (`Buy`) would require ~1.
*   Action 2 (`Sell`) would require ~2.

An uninitialized agent outputting ~0.0 would exclusively choose **Action 0 (Hold)** for thousands of steps, learning nothing.

### The Solution: Symmetric [-1, 1] Space
We define the action space as `Box(low=-1, high=1)`. We then map this continuous range to the discrete action indices.

**Formula:**
```python
scaled_value = ((raw_value + 1) / 2) * max_index
index = round(scaled_value)
```

This ensures that a raw output of **0.0** maps to the **middle** of the available actions, encouraging immediate interaction with the environment.

### Mapping Table (Standard 7 Actions)

For `max_index = 6` (Actions 0-6):

| Raw Input Range | Scaled Value | Action Index | Action Enum |
| :--- | :--- | :--- | :--- |
| `[-1.00, -0.83)` | `0` | **0** | `Hold` |
| `[-0.83, -0.50)` | `1` | **1** | `Buy` |
| `[-0.50, -0.17)` | `2` | **2** | `Sell` |
| `[-0.17, +0.17)` | `3` | **3** | `LimitBuy` (**Center**) |
| `[+0.17, +0.50)` | `4` | **4** | `LimitSell` |
| `[+0.50, +0.83)` | `5` | **5** | `StopLoss` |
| `[+0.83, +1.00]` | `6` | **6** | `TakeProfit` |

*Note: An uninitialized agent (0.0) defaults to **LimitBuy**, immediately attempting trades.*

### Amount Scaling
We apply the same logic to the **Amount** dimension.
*   Range: `[-1, 1]`
*   Mapping: `amount_pct = (raw_amount + 1) / 2`
*   Result: An output of 0.0 results in **50%** of available capital/shares being used, preventing 0-size orders.

### Time-In-Force (TIF) Scaling
For `TimeInForceActionStrategy`, the TIF dimension is also symmetric.
*   Range: `[-1, 1]` mapped to indices 0, 1, 2.
*   0.0 maps to **Index 1 (IOC)**.

| Raw Input Range | TIF Index | OrderTIF |
| :--- | :--- | :--- |
| `[-1.0, -0.33)` | 0 | `GTC` |
| `[-0.33, +0.33)` | 1 | `IOC` |
| `[+0.33, +1.0]` | 2 | `TTL` |

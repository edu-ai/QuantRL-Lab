# Multi-Asset Environment Examples

> **🚧 Coming Soon** — this folder is a placeholder for future multi-asset environment examples.

## What belongs here

Examples in this folder will use a **dedicated multi-asset environment** where the agent manages a *portfolio* of assets simultaneously — making joint allocation decisions (how much of each asset to hold) rather than trading each asset independently.

This is fundamentally different from the single-asset examples in `../single_asset/`:

| Aspect | `single_asset/` | `multi_asset/` (future) |
|---|---|---|
| Environment class | `SingleStockTradingEnv` | `MultiAssetTradingEnv` (TBD) |
| Agent action | Buy / Sell / Hold one stock | Allocate weights across N assets |
| Vectorization | N envs × 1 stock each | 1 env × N assets |
| Observation space | Per-stock features + portfolio state | Cross-asset features + covariance matrix |
| Reward signal | Per-stock P&L | Portfolio-level Sharpe / Sortino |

## Planned examples

- `portfolio_training.py` — PPO agent managing a 10-stock portfolio with position-size actions
- `mean_variance_baseline.py` — Markowitz benchmark to compare RL returns against
- `risk_parity_baseline.py` — Inverse-volatility baseline

## Related modules

- Alpha research for multi-asset: `src/quantrl_lab/alpha_research/`
- Future environment: `src/quantrl_lab/environments/portfolio/` (TBD)

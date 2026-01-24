# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

**Package manager:** uv (not Poetry)

```bash
# Install dependencies
uv sync                           # Core deps
uv sync --extra dev --extra notebooks  # With extras

# Run tests
uv run pytest                                # All tests
uv run pytest tests/data/                    # By module
uv run pytest tests/data/test_indicators.py  # Single file
uv run pytest -k "test_portfolio"            # By pattern
uv run pytest --cov=quantrl_lab              # With coverage
uv run pytest -m "not integration"           # Skip integration tests (for CI/CD)

# Code quality (auto-runs on commit)
pre-commit run --all-files        # Manual run

# Activate venv
source .venv/bin/activate
```

## Architecture

QuantRL-Lab is a reinforcement learning testbed for financial trading. The core design uses **dependency injection of pluggable strategies** to decouple environment logic from algorithmic choices.

### Strategy Pattern (Critical Concept)

The environment accepts 3 strategy objects at instantiation:

```python
env = SingleStockTradingEnv(
    data=df,
    config=config,
    action_strategy=action_strategy,      # How actions map to trades
    reward_strategy=reward_strategy,      # How rewards are calculated
    observation_strategy=observation_strategy  # What agent observes
)
```

This allows changing reward functions, observation features, or action spaces without modifying environment code.

### Data Flow

```
DataLoader (Alpaca/YFinance/AlphaVantage)
  → DataProcessor.apply_indicators()
  → SingleStockTradingEnv (with injected strategies)
  → RL Agent (PPO/SAC/A2C from stable-baselines3)
```

### Key Patterns

1. **Protocol-based data sources** - Data sources implement capability protocols (`HistoricalDataCapable`, `LiveDataCapable`, etc.) for runtime feature detection
2. **Indicator Registry** - Technical indicators auto-registered via `@IndicatorRegistry.register('NAME')` decorator
3. **Weighted Composite Rewards** - Combine multiple reward components with configurable weights

## Code Style

- Line length: 120 chars
- Black formatter with `--skip-string-normalization`
- isort with black profile
- flake8 for linting
- Python 3.10+ required
- **Docstrings:** Google-style format (autoDocstring compatible)

```python
def example_function(param1: str, param2: int = 10) -> pd.DataFrame:
    """
    Brief description of the function.

    Args:
        param1 (str): Description of param1.
        param2 (int, optional): Description of param2. Defaults to 10.

    Returns:
        pd.DataFrame: Description of return value.

    Raises:
        ValueError: When something is invalid.
    """
```

## Key Entry Points

| Task | Location |
|------|----------|
| Main environment | `src/quantrl_lab/environments/stock/env_single_stock.py` |
| Reward strategies | `src/quantrl_lab/environments/stock/strategies/rewards/` |
| Action strategies | `src/quantrl_lab/environments/stock/strategies/actions/` |
| Observation strategies | `src/quantrl_lab/environments/stock/strategies/observations/` |
| Data sources | `src/quantrl_lab/data/sources/` |
| Indicator registry | `src/quantrl_lab/data/indicators/` |
| Backtesting runner | `src/quantrl_lab/experiments/backtesting/runner.py` |
| Data processor | `src/quantrl_lab/data/processors/processor.py` |

## Environment Variables

Requires `.env` file (copy from `.env.example`):
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` - Market data & trading
- `ALPHA_VANTAGE_API_KEY` - Alternative data source (free tier: 25 req/day, intraday requires premium)
- `OPENAI_API_KEY` - For LLM hedge screener (optional)

## Non-Obvious Behaviors

- `current_step` is 0-indexed into the data array
- `window_size` determines lookback for observations
- Portfolio resets to `initial_balance` on `reset()` but keeps transaction history
- Price column auto-detected: searches 'close', 'Close', 'adj_close', or 4th column

## Step Execution Order

```python
# Inside env.step(action):
1. action_strategy.process_action(action)  # Validate & execute trades
2. portfolio.update_holdings()             # Update positions
3. observation = observation_strategy.build_observation()
4. reward = reward_strategy.calculate_reward()
5. done = check_terminal_conditions()
```

## Test Structure

Tests mirror the source structure under `tests/`:
```
tests/
├── conftest.py                 # Shared fixtures
├── data/
│   ├── test_indicators.py      # Technical indicator tests
│   ├── test_data_sources.py    # Unit tests (mocked APIs)
│   └── test_data_sources_integration.py  # Integration tests (real APIs)
└── environments/stock/
    ├── test_env.py             # Environment tests
    ├── test_portfolio.py       # Portfolio tests
    ├── test_action.py          # Action strategy tests
    └── test_reward.py          # Reward strategy tests
```

Integration tests require API keys in `.env` and are marked with `@pytest.mark.integration`.

## See Also

- `AGENTS.md` - Comprehensive architecture guide with project structure
- `docs/ARCHITECTURE.md` - Detailed diagrams and design patterns
- `notebooks/backtesting_example.ipynb` - Full workflow example

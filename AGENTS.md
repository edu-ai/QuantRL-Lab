# AGENTS.md

This file provides guidance to agents (i.e., ADAL) when working with code in this repository.

## Essential Commands

### Package Management
**This project uses uv for dependency management.**

```bash
# Install all dependencies
uv sync

# Install with specific extras
uv sync --extra dev --extra notebooks

# Activate virtual environment
source .venv/bin/activate

# Add a dependency
uv add package-name

# Add a dev dependency
uv add --group dev package-name
```

### Testing
```bash
# Run all tests
uv run pytest

# Run by module
uv run pytest tests/data/
uv run pytest tests/environments/stock/

# Run specific test file
uv run pytest tests/data/test_indicators.py

# Run with coverage
uv run pytest --cov=quantrl_lab

# Run tests matching pattern
uv run pytest -k "test_portfolio"

# Skip integration tests (for CI/CD - no API keys)
uv run pytest -m "not integration"

# Run only integration tests (requires .env with API keys)
uv run pytest -m integration
```

### Code Quality (Pre-commit Hooks)
**Pre-commit hooks automatically run on `git commit`** - checking formatting, linting, etc.

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip hooks temporarily
git commit -m "message" --no-verify
```

**Hooks configuration:** `.pre-commit-config.yaml`
- Black (formatting, line-length=120, skip string normalization)
- isort (import sorting, black profile)
- flake8 (linting, max-line-length=120)
- docformatter (docstring formatting)

### Notebooks
```bash
# Install Jupyter kernel (one-time)
python -m ipykernel install --user --name quantrl-lab --display-name "QuantRL-Lab"

# Start Jupyter
jupyter notebook
# Then select "QuantRL-Lab" kernel
```

**Key notebooks:**
- `notebooks/backtesting_example.ipynb` - Main workflow demo
- `notebooks/feature_selection.ipynb` - Feature engineering
- `notebooks/hyperparameter_tuning.ipynb` - Optuna tuning

### Development Gotchas
- **Never commit `.env`** - contains API keys (Alpaca, Alpha Vantage, etc.)
- **Use `.env.example` as template** for required environment variables
- **Python 3.10+ required** (see pyproject.toml)
- **All imports use `src.quantrl_lab.*`** - package installed in editable mode via Poetry

## Architecture Overview

### Core Design Pattern: Strategy Injection

**Critical:** This framework decouples environment logic from policies via **dependency injection of strategies**. The environment accepts 3 pluggable strategy objects at instantiation:

```python
env = SingleStockTradingEnv(
    data=df,
    config=config,
    action_strategy=action_strategy,      # How actions are processed
    reward_strategy=reward_strategy,      # How rewards are calculated
    observation_strategy=observation_strategy  # What the agent observes
)
```

**Why this matters:**
- Change reward functions WITHOUT touching environment code
- Swap observation features WITHOUT rewriting state logic
- Experiment with different action spaces by injecting new strategies
- Compose complex behaviors from simple, reusable components

### Key Integration Points

**1. Data Flow: Source → Processor → Environment**

```
DataLoader (Alpaca/YFinance/AlphaVantage/FMP)
  ↓ fetch_data()
DataFrame with OHLCV
  ↓ DataProcessor.apply_indicators()
DataFrame with technical indicators
  ↓ pass to env
SingleStockTradingEnv
  ↓ step() delegates to strategies
Action/Observation/Reward strategies
```

**2. Protocol-Based Data Sources**

Data sources inherit `DataSource` base class and implement capability protocols:
- `HistoricalDataCapable` - historical OHLCV
- `LiveDataCapable` - real-time quotes
- `NewsDataCapable` - news sentiment
- `StreamingCapable` - websocket streaming

**Example:**
```python
# Check capabilities at runtime
if isinstance(loader, LiveDataCapable):
    quote = loader.get_latest_quote(symbol)
```

See `src/quantrl_lab/data/interface.py` for protocol definitions.

**3. Indicator Registry Pattern**

Technical indicators are auto-registered via decorator:

```python
@IndicatorRegistry.register('RSI')
def rsi(df, window=14):
    # calculation
    return df

# Apply dynamically
df = IndicatorRegistry.apply('RSI', df, window=14)

# List all available
IndicatorRegistry.list_all()  # ['SMA', 'EMA', 'RSI', 'MACD', ...]
```

**Key file:** `src/quantrl_lab/data/indicators/registry.py`

**4. BacktestRunner Workflow**

```python
# 1. Create env config factory
env_config = BacktestRunner.create_env_config_factory(
    train_data=train_df,
    test_data=test_df,
    action_strategy=action_strategy,
    reward_strategy=reward_strategy,
    observation_strategy=observation_strategy,
    # ... other params
)

# 2. Run single experiment
runner = BacktestRunner(verbose=1)
results = runner.run_single_experiment(
    PPO,  # Algorithm class from stable-baselines3
    env_config,
    total_timesteps=50000,
    num_eval_episodes=3
)

# 3. Inspect results
BacktestRunner.inspect_single_experiment(results)

# 4. Or run comprehensive sweep
results = runner.run_comprehensive_backtest(
    algorithms=[PPO, SAC, A2C],
    env_configs=configs,
    presets=["conservative", "explorative"],
    total_timesteps=50000
)
```

**Entry point:** `src/quantrl_lab/experiments/backtesting/runner.py`

### Project Structure

```
src/quantrl_lab/
├── environments/              # Gymnasium trading environments (renamed from custom_envs)
│   ├── base/                 # Base classes (TradingEnv, Portfolio, Config) - renamed from core
│   ├── strategies/           # Shared strategy interfaces (NEW - extracted from stock/)
│   │   ├── actions.py       # BaseActionStrategy
│   │   ├── observations.py  # BaseObservationStrategy
│   │   └── rewards.py       # BaseRewardStrategy
│   │
│   └── stock/               # Single-stock implementation
│       ├── env_single_stock.py         # Main environment
│       ├── stock_portfolio.py          # Portfolio state management
│       └── strategies/      # Stock-specific strategy implementations
│           ├── actions/     # Action space definitions
│           ├── observations/  # State representation
│           └── rewards/     # Reward functions
│
├── data/                     # Data acquisition & processing
│   ├── sources/             # Data source loaders
│   │   ├── alpaca_loader.py
│   │   ├── yfinance_loader.py
│   │   ├── alpha_vantage_loader.py
│   │   └── fmp_loader.py
│   │
│   │
│   ├── processors/          # Data transformation (NEW - separated from sources)
│   │   ├── processor.py    # DataProcessor (was data_processor.py)
│   │   └── mappings/       # API response normalization
│   │
│   └── indicators/          # Technical indicator registry + implementations
│
├── experiments/            # Offline experimentation (NEW top-level grouping)
│   ├── backtesting/        # Training & evaluation orchestration (moved from top-level)
│   │   ├── runner.py       # Main entry point
│   │   ├── training.py     # Model training logic
│   │   ├── evaluation.py   # Performance metrics
│   │   └── config/         # Algorithm configs & presets
│   │
│   ├── feature_engineering/  # Vectorized backtesting (renamed from feature_selection/)
│   │   ├── analyzer.py     # Indicator analysis
│   │   └── vectorized/     # Vectorized strategy implementations
│   │
│   └── tuning/             # Optuna hyperparameter optimization (moved from top-level)
│
├── experiments/            # Offline experimentation (NEW top-level grouping)
│   ├── backtesting/        # Training & evaluation orchestration (moved from top-level)
│   │   ├── runner.py       # Main entry point
│   │   ├── training.py     # Model training logic
│   │   ├── evaluation.py   # Performance metrics
│   │   └── config/         # Algorithm configs & presets
│   │
│   ├── feature_engineering/  # Vectorized backtesting (renamed from feature_selection/)
│   │   ├── analyzer.py     # Indicator analysis
│   │   └── vectorized/     # Vectorized strategy implementations
│   │
│   ├── tuning/             # Optuna hyperparameter optimization (moved from top-level)
│   │
│   └── screening/          # LLM-based hedge pair screening (for pair discovery)
│       ├── llm_hedge_screener.py
│       ├── response_schemas.py
│       ├── data_models.py
│       └── prompt.py
│
├── deployment/             # Production workflows (NEW top-level grouping)
│   └── trading/           # Live trading with Alpaca (moved from top-level)
│
└── utils/                 # Shared utilities

tests/                     # Pytest test suite (mirrors src/ structure)
│   ├── data/             # Data module tests
│   │   ├── test_indicators.py
│   │   ├── test_data_sources.py
│   │   └── test_data_sources_integration.py
│   └── environments/stock/  # Environment tests
│
notebooks/                 # Usage examples
```

## Common Workflows

### Adding a New Technical Indicator

1. Add function to `src/quantrl_lab/data/indicators/technical.py`
2. Decorate with `@IndicatorRegistry.register('INDICATOR_NAME')`
3. Function signature: `def indicator_name(df: pd.DataFrame, **kwargs) -> pd.DataFrame`
4. It's auto-discoverable via `IndicatorRegistry.list_all()`

### Creating a Custom Reward Strategy

1. Inherit `BaseRewardStrategy` from `src/quantrl_lab/environments/strategies/rewards.py`
2. Implement `calculate_reward(self) -> float` (reads from `self.env`)
3. Optionally implement `reset()` for episode initialization
4. Inject into environment via `reward_strategy` parameter

**Example:** See `src/quantrl_lab/environments/stock/strategies/rewards/trend_following_reward.py`

### Running Experiments

**Typical flow:**
1. Load data with DataLoader
2. Process features with DataProcessor
3. Define strategies (action/observation/reward)
4. Create env config factory
5. Run BacktestRunner
6. Analyze results

**See:** `notebooks/backtesting_example.ipynb` for complete example

### Updating Documentation

**When making API changes, ALWAYS update documentation:**

1. **API endpoint changes** - module renames, class moves, new public APIs
2. **Update MkDocs references** in `docs/api-reference/` to match actual module paths
3. **Verify build** - run `mkdocs build` to catch errors before committing

**Common doc files to check:**
- `docs/api-reference/data-sources.md` - data loader modules
- `docs/api-reference/strategies.md` - strategy interfaces and implementations
- `docs/api-reference/environments.md` - environment classes

**Example:** If you rename `alpaca.py` → `alpaca_loader.py`, update docs from:
```
::: quantrl_lab.data.sources.alpaca
```
to:
```
::: quantrl_lab.data.sources.alpaca_loader
```

**Always run `mkdocs build` before committing to ensure docs build without errors.**

## Non-Obvious Behaviors

### Environment State Management

- **`current_step` is 0-indexed** - points to current row in data array
- **`window_size` lookback** - observations include past N steps
- **Portfolio resets to initial_balance** on `reset()`, but keeps transaction history for analysis
- **Price column auto-detection** - searches for 'close', 'Close', 'adj_close', or 4th column if DataFrame

### Strategy Execution Order (per step)

```python
# Inside env.step(action):
1. action_strategy.process_action(action)  # Validate & execute trades
2. portfolio.update_holdings()             # Update positions
3. observation = observation_strategy.build_observation()  # Get state
4. reward = reward_strategy.calculate_reward()  # Calculate reward
5. done = check_terminal_conditions()      # Episode termination
```

### WeightedCompositeReward

**Combines multiple reward components with configurable weights:**

```python
reward_strategy = WeightedCompositeReward(
    components=[
        PortfolioValueChangeReward(),
        TrendFollowingReward(),
        InvalidActionPenalty(),
    ],
    weights=[0.5, 0.3, 0.2]  # Must sum to 1.0
)
```

**Presets available:** "conservative", "explorative", "balanced", "risk_managed"
(see `src/quantrl_lab/experiments/backtesting/config/reward_presets.py`)

### Data Source Capabilities

**Not all data sources support all features:**
- Alpaca: Historical, Live, Streaming, News (requires API keys)
- YFinance: Historical, Fundamentals (free, no key needed)
- AlphaVantage: Historical, Fundamentals, Macroeconomic, News (requires API key)

**Alpha Vantage Free Tier Limitations:**
- 25 requests/day, 1 request/second burst limit
- `outputsize=full` (20+ years of data) requires premium
- Intraday data (1min, 5min, etc.) requires premium
- The loader auto-handles rate limiting and defaults to `compact` (last 100 days)

**Check capabilities before use:**
```python
features = loader.supported_features()
# Returns: {'historical': True, 'live': False, ...}
```

## Environment Variables

**Required in `.env` (copy from `.env.example`):**

```bash
# Alpaca Trading API (for data & live trading)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Alpha Vantage (for fundamentals, macro data, news)
# Free tier: 25 req/day, intraday & outputsize=full require premium
ALPHA_VANTAGE_API_KEY=your_key

# Optional: LLM APIs for hedge screener
OPENAI_API_KEY=your_key
```

## Testing Strategy

**Test structure mirrors source code:**
```
tests/
├── conftest.py                              # Shared fixtures
├── data/
│   ├── test_indicators.py                   # IndicatorRegistry & technical indicators
│   ├── test_data_sources.py                 # Unit tests with mocked APIs
│   └── test_data_sources_integration.py     # Integration tests with real APIs
└── environments/stock/
    ├── test_env.py                          # Environment step logic, terminal conditions
    ├── test_portfolio.py                    # Position tracking, cash flow
    ├── test_action.py                       # Action strategy validation
    └── test_reward.py                       # Reward calculation correctness
```

**Test categories:**
- **Unit tests** - Fast, mocked, run in CI/CD
- **Integration tests** - Real API calls, require `.env`, marked with `@pytest.mark.integration`

**Run before commits:**
```bash
pre-commit run --all-files       # Formatting + linting
uv run pytest -m "not integration"  # Unit tests only
uv run pytest                    # All tests (if API keys available)
```

## Key Files to Check First

**When modifying:**
- Action spaces → `src/quantrl_lab/environments/stock/strategies/actions/`
- Reward logic → `src/quantrl_lab/environments/stock/strategies/rewards/`
- Data loading → `src/quantrl_lab/data/sources/`
- Feature engineering → `src/quantrl_lab/data/processors/processor.py`
- Backtesting flow → `src/quantrl_lab/experiments/backtesting/runner.py`

**When debugging:**
- Check `StockPortfolio` state in `src/quantrl_lab/environments/stock/stock_portfolio.py`
- Verify data array shape and `current_step` index
- Confirm strategy injection in environment `__init__`
- Review `BacktestRunner.inspect_single_experiment()` output

## External Dependencies

**Core ML stack:**
- `stable-baselines3` - PPO, SAC, A2C algorithms
- `gymnasium` - RL environment interface
- `optuna` - Hyperparameter tuning

**Data & analysis:**
- `pandas`, `numpy` - Data manipulation
- `yfinance`, `alpaca-py` - Market data
- `seaborn`, `matplotlib` - Visualization

**See `pyproject.toml` for complete list.**

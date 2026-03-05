# QuantRL-Lab Architecture Guide

This document covers the key architectural decisions and design patterns in QuantRL-Lab. For practical usage, see the [User Guide](user-guide/overview.md).

## Table of Contents
- [System Layers](#system-layers)
- [Strategy Pattern (Dependency Injection)](#strategy-pattern-dependency-injection)
- [Step Execution Order](#step-execution-order)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Protocol Pattern for Data Sources](#protocol-pattern-for-data-sources)
- [Registry Pattern for Technical Indicators](#registry-pattern-for-technical-indicators)
- [Non-Obvious Behaviours](#non-obvious-behaviours)

---

## System Layers

QuantRL-Lab is organized into four layers:

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **Data** | Fetch, normalize, and engineer features from market data | `DataSource`, `DataProcessor`, `DataPipeline` |
| **Environment** | Gymnasium-compatible trading simulation | `SingleStockTradingEnv`, pluggable strategies |
| **Experiment** | Train, tune, and evaluate RL agents | `BacktestRunner`, `ExperimentJob`, `OptunaRunner` |
| **Utilities** | Feature selection, logging | `IndicatorRegistry`, `AgentExplainer` |

Data flows top-to-bottom: raw market data → processed features → environment → RL agent → evaluation results.

---

## Strategy Pattern (Dependency Injection)

The core design principle. `SingleStockTradingEnv` accepts three pluggable strategies at construction time — it never hardcodes how actions, observations, or rewards work:

```python
env = SingleStockTradingEnv(
    data=df,
    config=config,
    action_strategy=...,       # How raw agent output maps to market orders
    observation_strategy=...,  # What state the agent sees
    reward_strategy=...,       # What scalar signal the agent optimizes
)
```

Each strategy type has an abstract base class in `quantrl_lab.environments.core.interfaces`:

**`BaseActionStrategy`**
```python
def define_action_space(self) -> gym.spaces.Space: ...
def handle_action(self, env: TradingEnvProtocol, action: Any) -> Tuple[Any, Dict]: ...
```

**`BaseObservationStrategy`**
```python
def define_observation_space(self, env: TradingEnvProtocol) -> gym.spaces.Space: ...
def build_observation(self, env: TradingEnvProtocol) -> np.ndarray: ...
def get_feature_names(self, env: TradingEnvProtocol) -> List[str]: ...
```

**`BaseRewardStrategy`**
```python
def calculate_reward(self, env: TradingEnvProtocol) -> float: ...
def on_step_end(self, env: TradingEnvProtocol): ...  # optional hook for stateful strategies
```

Strategies are injected, not inherited. The environment calls `handle_action()`, `calculate_reward()`, and `build_observation()` each step — it doesn't care about the implementation. This makes it trivial to swap reward functions without touching environment code.

---

## Step Execution Order

Each `env.step(action)` call follows this exact sequence (see `single.py`):

1. Store `prev_portfolio_value` (for reward calculations)
2. `portfolio.process_open_orders()` — fill or expire pending limit/stop orders
3. `action_strategy.handle_action(env, action)` — decode and execute the new order
4. Advance `current_step`, check `terminated` / `truncated`
5. `reward_strategy.calculate_reward(env)` — compute scalar reward
6. Clip reward to `reward_clip_range`
7. `reward_strategy.on_step_end(env)` — stateful hook (e.g. update running stats)
8. `observation_strategy.build_observation(env)` — construct next state vector
9. Return `(observation, reward, terminated, truncated, info)`

Note: reward is computed **before** observation. Both strategies receive the environment instance (`env`) so they can access `env.portfolio`, `env.data`, `env.current_step`, `env.action_type`, etc.

---

## Data Processing Pipeline

Raw OHLCV data passes through a **builder-pattern pipeline** (`DataPipeline`) composed of `ProcessingStep` units:

```
DataPipeline
├── TechnicalIndicatorStep   (SMA, EMA, RSI, MACD, ...)
├── AnalystEstimatesStep     (price targets, ratings — requires FMP API key)
├── MarketContextStep        (sector/industry performance)
├── SentimentEnrichmentStep  (news sentiment via HuggingFace — optional)
├── NumericConversionStep    (cast columns to float32)
└── ColumnCleanupStep        (drop non-numeric, rename)
```

`DataProcessor.data_processing_pipeline()` wraps this into a single call. All steps are tracked in the returned `ProcessingMetadata` object (which columns were dropped, which indicators were added, etc).

`ProcessingStep` is a **Protocol** (structural typing) rather than an ABC — any class with a `process(df, metadata)` method qualifies. This makes it easy to add custom steps without modifying the pipeline itself.

---

## Protocol Pattern for Data Sources

Data sources use a hybrid of ABC + Protocols. `DataSource` (ABC) provides shared infrastructure (`source_name`, `connect()`, `supported_features`). Optional capabilities are expressed as runtime-checkable Protocols — a loader satisfies a protocol simply by having the required methods, with no inheritance needed:

| Protocol | Key Methods |
|----------|-------------|
| `HistoricalDataCapable` | `get_historical_ohlcv_data()` |
| `LiveDataCapable` | `get_latest_quote()`, `get_latest_trade()` |
| `StreamingCapable` | `subscribe_to_updates()`, `start_streaming()`, `stop_streaming()` |
| `NewsDataCapable` | `get_news_data()` |
| `FundamentalDataCapable` | `get_fundamental_data()` |
| `AnalystDataCapable` | `get_historical_grades()`, `get_historical_rating()` |
| `SectorDataCapable` | `get_historical_sector_performance()`, `get_historical_industry_performance()` |
| `CompanyProfileCapable` | `get_company_profile()` |

This avoids forcing every loader to inherit from a sprawling ABC chain. Adding a new capability means defining a new protocol and implementing the method on the relevant loaders — the class hierarchy doesn't change. Capabilities are checked at runtime via `isinstance(loader, SomeProtocol)`, and `supported_features` auto-discovers which protocols a loader implements.

---

## Registry Pattern for Technical Indicators

Technical indicators are registered via a decorator on module import, eliminating hardcoded lists:

```python
@IndicatorRegistry.register("RSI")
def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    ...
```

The registry provides:
- `IndicatorRegistry.list_all()` — discover all available indicators dynamically
- `IndicatorRegistry.apply("RSI", df, window=14)` — apply any indicator by name

This is what makes the feature selection workflow (`notebooks/feature_selection.ipynb`) possible: the notebook can iterate over all registered indicators programmatically, compute them all, and rank by correlation with returns — without maintaining a hardcoded list anywhere.

Adding a new indicator is a one-step operation: decorate the function with `@IndicatorRegistry.register("NAME")`. It becomes immediately available to `DataProcessor` and the feature selection pipeline.

---

## Non-Obvious Behaviours

Things that aren't apparent from the API surface and tend to cause confusion:

- **`current_step` advances after the action, not before reward.** Reward is computed on the post-action state at the new `current_step`, so `env.data[env.current_step]` inside `calculate_reward()` is the bar *after* the trade was placed, not the bar it was placed on.

- **`action_type` is set by `handle_action()` and read by reward strategies.** If your reward strategy inspects `env.action_type`, it will always see the *current* step's action — `handle_action()` stores it on `self` before `calculate_reward()` is called.

- **Portfolio resets on `reset()` but retains no episode history.** `env.reset()` resets balance, shares, and open orders to initial state. The portfolio's transaction log is also cleared. Episode data is only preserved if you collect it externally (e.g. via `BacktestRunner`).

- **Price column is auto-detected.** The environment searches for columns named `close`, `Close`, or `adj_close`; falls back to the 4th column (index 3) if none match. Pass `price_column=` explicitly to override.

- **`window_size` affects observation padding, not data slicing.** The full dataset is always available via `env.data`. `window_size` controls how many bars the observation strategy uses for its rolling window, with zero-padding at the start of an episode.

- **`CompositeReward` weights are not automatically normalised.** Pass `normalize_weights=True` to normalise them to sum to 1. `auto_scale=True` is separate — it standardises each component to N(0,1) before weighting, which is generally recommended when mixing strategies with different magnitudes.

- **`n_envs > 1` in `ExperimentJob` uses vectorised environments for training only.** Evaluation always runs on a single environment regardless of `n_envs`.

- **Limit and stop orders persist across steps.** An unexecuted order placed at step `t` stays in `portfolio.open_orders` and is checked at the start of every subsequent step via `process_open_orders()` until it fills or the episode ends. Orders placed with `OrderTIF.TTL` additionally expire after `SimulationConfig.order_expiration_steps` steps (default 5).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-05

### Added

- **New reward strategies** (`src/quantrl_lab/environments/stock/strategies/rewards/`):
  - `BoredomPenaltyReward`: Penalizes holding an open position for too long to discourage passive strategies.
  - `LimitExecutionReward`: Rewards price improvement achieved when a limit order fills better than the reference price.
  - `TurnoverPenaltyReward`: Penalizes excessive trading proportional to fees paid, encouraging efficiency.
  - `OrderExpirationPenaltyReward`: Penalizes expired limit orders to discourage order spamming.
- **`BacktestEnvironmentBuilder`** (`src/quantrl_lab/experiments/backtesting/builder.py`):
  - Fluent builder API replacing the deprecated `create_env_config_factory` function.
  - Supports `with_data()`, `with_env_params()`, `with_strategies()`, and `build()` chaining.
  - Accepts optional `eval_data` for 3-way train/eval/test splits used in Optuna tuning.
- **`ExperimentJob` and `JobGenerator`** (`src/quantrl_lab/experiments/backtesting/core.py`):
  - `ExperimentJob`: Dataclass encapsulating algorithm class, env config, hyperparameters, and timesteps.
  - `JobGenerator.generate_grid()`: Generates a cross-product batch of jobs across algorithms, configs, and env configs.
- **`AgentExplainer`** (`src/quantrl_lab/experiments/backtesting/explainer.py`):
  - Feature importance analysis via Input×Gradient saliency (deep learning attribution) with automatic fallback to Pearson correlation.
  - Supports RecurrentPPO/LSTM policies (forced correlation mode).
  - Always restores `training_mode` after saliency analysis via `try/finally`.
- **`BacktestMetrics`** (`src/quantrl_lab/experiments/backtesting/metrics.py`):
  - Standalone metrics computation: Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, win rate, profit factor.
- **Action distribution display** in `BacktestRunner.inspect_result()`: aggregates and prints a table of action counts and percentages across all episodes.
- **Async data loading**: All data loaders now expose `async_fetch_ohlcv()` and related async methods for concurrent multi-symbol fetching.
- **`async_request_utils`** (`src/quantrl_lab/data/utils/async_request_utils.py`): shared async HTTP helpers with retry and rate-limit handling.
- **Alpha research module** (`src/quantrl_lab/alpha_research/`):
  - `AlphaSelector`: selects informative indicators using IC (Information Coefficient), Rank IC, and win rate metrics.
  - `AlphaRegistry`: decorator-based registry for custom alpha strategies.
  - `AlphaEnsemble`: combines multiple alpha signals with configurable weighting.
  - `AlphaRunner`: orchestrates end-to-end signal discovery workflows.
  - Vectorized strategy implementations: `BollingerBandsStrategy`, momentum, mean-reversion.
  - Visualization helpers for signal analysis reports.
- **End-to-end example scripts** (`examples/end_to_end/`):
  - `train_single_symbol.py`: single-stock PPO training with `BacktestEnvironmentBuilder`.
  - `train_multi_symbol.py`: multi-symbol shared PPO policy via `SubprocVecEnv` + RecurrentPPO/LSTM.
  - `train_single_symbol_sac.py`: single-stock SAC (off-policy, `n_envs=1`, `ent_coef="auto"`).
  - `train_single_symbol_a2c.py`: single-stock A2C (`n_envs=4`, short rollout `n_steps=5`).
  - `train_multi_symbol_sac.py`: per-symbol SAC grid via `JobGenerator` + `inspect_batch()`.
  - `train_multi_symbol_a2c.py`: shared A2C policy across symbols via `SubprocVecEnv`.
  - `tune_single_symbol.py`: full Optuna PPO tuning with 3-way split, refit, and test evaluation.
  - Shared `data_utils.py` with `init_data_sources()`, `select_alpha_indicators()`, `process_symbol()`, and split helpers.
- **Optuna tuning improvements** (`src/quantrl_lab/experiments/tuning/optuna_runner.py`):
  - SQLite study persistence with `load_if_exists=True`.
  - Guard against unsafe `n_jobs > 1` with SQLite storage (auto-fallback to 1).
  - `create_ppo_search_space()` helper with recommended search ranges.
- **Alternative data pipeline steps** (`src/quantrl_lab/data/processing/steps/alternative/`):
  - `analyst.py`: integrates analyst ratings and grades into the feature pipeline.
  - `sentiment.py`: integrates news sentiment scores.
- **Test coverage expansions**:
  - `tests/environments/stock/strategies/rewards/test_boredom.py`
  - `tests/environments/stock/test_time_in_force_action.py`
  - `tests/experiments/backtesting/test_builder.py`, `test_metrics.py`, `test_runner.py`
  - `tests/experiments/tuning/test_optuna_runner.py`
  - `tests/data/sources/test_async_loaders.py`
  - `tests/data/utils/test_async_request_utils.py`
  - `tests/alpha_research/test_alpha_research.py`
  - `tests/data/processing/steps/alternative/test_analyst.py`
- **GitHub Pages documentation hosting** via `deploy-docs.yml` GitHub Actions workflow.
- **GEMINI.md** and **AGENTS.md**: guidance files for AI assistants with architecture patterns and common workflows.

### Changed

- **`BacktestRunner` API** (`src/quantrl_lab/experiments/backtesting/runner.py`):
  - Replaced `run_single_experiment()` / `inspect_single_experiment()` with `run_job()` / `inspect_result()`.
  - Added `run_batch()` / `inspect_batch()` for batch experiment comparison.
  - Fixed critical bug: `total_timesteps` from `ExperimentJob` was never forwarded to `train_model()` (always used default 10,000).
  - Initialized `explanation_method` before try block to prevent potential `UnboundLocalError`.
- **`DataPipeline` builder pattern** (`src/quantrl_lab/data/processing/pipeline.py`): refined step registration and async step execution support.
- **`FeatureAwareObservationStrategy`** (`src/quantrl_lab/environments/stock/strategies/observations/feature_aware.py`): improved stationary feature normalization logic.
- **Reward shaping guidance**: all example scripts now use `PortfolioValueChangeReward` as the primary signal with a minimal `TurnoverPenaltyReward`. Heavy penalties (Sortino, drawdown) are documented as causing "do nothing" convergence in short training runs.
- **Documentation overhaul** (`docs/`):
  - Added standalone top-level guide tabs: `DATA_SOURCES.md`, `data-processing.md`, `environments.md`, `experiments.md`.
  - Slimmed `ARCHITECTURE.md` to ~170 lines; removed Mermaid diagrams that caused rendering issues.
  - Eliminated duplicated content across `overview.md`, `configuration.md`, `backtesting.md`, and `environments.md` — replaced with cross-links.
  - GitHub Pages deployment switched to `uv sync --all-extras` so mkdocstrings can import the full package.
- **`experiments/alpha_research/` moved** to top-level `src/quantrl_lab/alpha_research/` and `experiments/__init__.py` updated accordingly.

### Fixed

- **`DifferentialSortinoReward` gradient explosion**: clamped intermediate values to prevent NaN/Inf rewards during training.
- **PPO scaling issue**: corrected observation and reward scaling so the agent makes meaningful trades instead of converging to zero-action policies.
- **`suggest_discrete_uniform` deprecation** in `optuna_runner.py`: migrated to `suggest_float(..., step=q)` (Optuna 3.x).
- **`AgentExplainer` training mode leak**: policy was left in eval mode after saliency analysis if an exception occurred mid-episode; fixed with `try/finally`.
- **`API Consistency`**: Fixed `execute_limit_order` → `place_limit_order` mismatch in `StockPortfolio`.
- **Critical execution logic** (`SingleStockTradingEnv`, `StockPortfolio`):
  - OHLC auto-detection: environment now reads `High`, `Low`, and `Open` columns for realistic simulation.
  - Limit/stop triggers now check `High` (sells) and `Low` (buys/stops) instead of only `Close`.
  - Gap handling: stop-loss executes at `Open` when price gaps down past the trigger.
- **YFinance loader**: fixed `period=timeframe` → `interval=timeframe` parameter causing API errors.
- **Alpha Vantage free tier**: changed default `outputsize` to `compact`, added 1.2s rate limiting, surfaced `Information` key errors.
- **Documentation build**: fixed 21 griffe docstring warnings, broken ARCHITECTURE.md links, and social plugin logo warnings.

### Refactor

- **Environment architecture** (`src/quantrl_lab/environments/`):
  - Created `environments/core/` consolidating base protocols (`interfaces.py`), types (`types.py`), portfolio (`portfolio.py`), and config.
  - Modularized `environments/stock/` into `components/` (portfolio, config) and `strategies/` (actions, observations, rewards).
  - Renamed `env_single_stock.py` → `single.py`; removed fragmented `environments/base/` and `environments/strategies/`.
- **Portfolio**: refactored `StockPortfolio` to use typed `Order` dataclasses and `OrderType` enums instead of plain dicts.
- **Configuration**: introduced `SimulationConfig` (market mechanics) and `RewardConfig` nested within `SingleStockEnvConfig`; tightened reward clipping from `[-5, 5]` → `[-1, 1]`.
- **Action strategy**: renamed `StandardMarketActionStrategy` → `StandardActionStrategy`.
- **Data utilities** (`src/quantrl_lab/data/utils/`): extracted date parsing, symbol handling, DataFrame normalization, response validation, and HTTP retry logic into dedicated modules (~1,000 lines, 118 tests), eliminating duplication across all loaders.
- **Data source naming**: renamed all loader files to `*_loader.py` suffix to prevent import conflicts with upstream packages (`alpaca-py`, `yfinance`).
- **Alpha research module**: renamed internal files to match project conventions (`base.py`, `strategies.py`); fixed duplicate fields in `AlphaJob` dataclass.
- **Examples cleanup**: removed broken/redundant `examples/pipelines/auto_indicator_selection.py`; consolidated shared logic into `examples/end_to_end/shared/data_utils.py`.

## [0.1.0] - 2025-10-16

### Added
- Initial release of QuantRL-Lab
- Gymnasium-compatible trading environments for stocks, crypto, and forex
- Modular strategy system with customizable actions, rewards, and observations
- Integration with Stable-Baselines3 and other RL frameworks
- Comprehensive backtesting framework with performance metrics
- Technical indicators library for feature engineering
- Support for multiple data sources (yfinance, Alpaca, Alpha Vantage)
- Optuna integration for hyperparameter tuning
- LLM-based hedge screening capabilities
- Complete CI/CD pipeline for automated testing and PyPI publishing
- Extensive documentation and examples

### Documentation
- Setup guide for development environment
- Publishing workflow documentation
- Example notebooks for backtesting, feature selection, and tuning
- API documentation

[Unreleased]: https://github.com/whanyu1212/QuantRL-Lab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/whanyu1212/QuantRL-Lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/whanyu1212/QuantRL-Lab/releases/tag/v0.1.0

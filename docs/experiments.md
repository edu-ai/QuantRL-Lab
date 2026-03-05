# Experiments

The `experiments` module provides a structured, reproducible workflow for training RL agents, evaluating them on held-out data, and tuning hyperparameters. It sits on top of the environment layer and handles the boilerplate of vectorized training, evaluation, metrics, and reporting.

---

## Module Overview

```
experiments/
├── backtesting/
│   ├── core.py          # ExperimentJob, ExperimentResult, JobGenerator
│   ├── runner.py        # BacktestRunner — orchestrates train + evaluate
│   ├── builder.py       # BacktestEnvironmentBuilder — fluent env factory
│   ├── metrics.py       # MetricsCalculator — financial + RL metrics
│   ├── explainer.py     # AgentExplainer — feature importance analysis
│   ├── training.py      # train_model() helper
│   └── config/
│       └── environment_config.py  # BacktestEnvironmentConfig
└── tuning/
    └── optuna_runner.py  # OptunaRunner + pre-built search spaces
```

---

## Core Concepts

The experiment workflow is built around three objects:

| Object | Role |
|---|---|
| `BacktestEnvironmentConfig` | Holds callable factories that create train/test envs |
| `ExperimentJob` | Bundles an algorithm class + env config + run params into one reproducible unit |
| `ExperimentResult` | Returned by the runner: metrics, trained model, episode data, feature importance |

The `BacktestRunner` takes a job, trains the agent, evaluates on both train and test splits, computes metrics, and runs feature importance analysis — all in one call.

---

## Building Environment Configs

### With `BacktestEnvironmentBuilder` (recommended)

The fluent builder is the cleanest way to create env configs. It handles data processing, strategy wiring, and factory creation.

```python
from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder
from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward

env_config = (
    BacktestEnvironmentBuilder()
    .with_data(train_data=train_df, test_data=test_df)
    .with_strategies(
        action=StandardActionStrategy(),
        reward=CompositeReward(
            strategies=[DifferentialSortinoReward(), DrawdownPenaltyReward()],
            weights=[0.7, 0.3],
        ),
        observation=FeatureAwareObservationStrategy(),
    )
    .with_env_params(
        initial_balance=100_000,
        window_size=20,
        transaction_cost_pct=0.001,
        slippage=0.0005,
    )
    .build()
)
```

The builder also accepts a custom `name` via `BacktestEnvironmentConfig` — useful for identifying results in batch runs:

```python
from quantrl_lab.experiments.backtesting.config.environment_config import BacktestEnvironmentConfig

config = env_config
config.name = "AAPL_Sortino_w20"
```

### With `BacktestRunner.create_env_config` (manual factories)

Use this when you need direct control over the factory functions:

```python
from quantrl_lab.experiments.backtesting.runner import BacktestRunner
from quantrl_lab.environments.stock.single import SingleStockTradingEnv
from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig

base_config = SingleStockEnvConfig(initial_balance=100_000, window_size=20)

env_config = BacktestRunner.create_env_config(
    train_env_factory=lambda: SingleStockTradingEnv(
        data=train_df, config=base_config,
        action_strategy=StandardActionStrategy(),
        reward_strategy=DifferentialSortinoReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    ),
    test_env_factory=lambda: SingleStockTradingEnv(
        data=test_df, config=base_config,
        action_strategy=StandardActionStrategy(),
        reward_strategy=DifferentialSortinoReward(),
        observation_strategy=FeatureAwareObservationStrategy(),
    ),
)
```

---

## Running a Single Job

```python
from stable_baselines3 import PPO
from quantrl_lab.experiments.backtesting.core import ExperimentJob
from quantrl_lab.experiments.backtesting.runner import BacktestRunner

job = ExperimentJob(
    algorithm_class=PPO,
    env_config=env_config,
    total_timesteps=100_000,
    n_envs=4,               # parallel envs used during training only
    num_eval_episodes=5,    # episodes run for train + test evaluation
    tags={"symbol": "AAPL", "reward": "sortino"},
)

runner = BacktestRunner(verbose=True)
result = runner.run_job(job)
```

The runner prints live progress via Rich:
```
══════════════════════════════════════════════════════════
RUNNING JOB: a3f1b2c4
Algo: PPO | Env: default_env
🔄 Training...
📊 Evaluating...
🧠 Analyzing Feature Importance...
Result:
  Train: +12.34% (Sharpe: 1.21)
  Test:   +7.89% (Sharpe: 0.94)
```

### Inspecting results

```python
BacktestRunner.inspect_result(result)
```

Prints a Rich panel with:
- Job ID, algorithm, env name, status
- Train / test average return and Sharpe
- Max drawdown
- Top learned features with importance scores
- Per-episode breakdown table (return %, reward, final value, steps)

### Accessing result data directly

```python
# Metrics dict (train_ and test_ prefixed)
print(result.metrics)
# {
#   'train_avg_return_pct': 12.34,
#   'train_avg_sharpe_ratio': 1.21,
#   'test_avg_return_pct': 7.89,
#   'test_avg_sharpe_ratio': 0.94,
#   'test_avg_max_drawdown': -0.08,
#   'test_win_rate': 0.6,
#   ...
# }

# Trained model (stable-baselines3 object)
model = result.model
obs, _ = test_env.reset()
action, _ = model.predict(obs, deterministic=True)

# Top features (name → importance score)
print(result.top_features)
# {'RSI_14_t-0': 0.42, 'portfolio_balance_ratio': -0.31, ...}

# Raw episode data
for ep in result.test_episodes:
    print(ep["final_value"], ep["total_reward"], ep["steps"])
```

---

## Running a Batch of Jobs

### Manual batch

```python
from stable_baselines3 import PPO, SAC, A2C

jobs = [
    ExperimentJob(PPO, env_config_aapl, total_timesteps=100_000, tags={"symbol": "AAPL"}),
    ExperimentJob(SAC, env_config_aapl, total_timesteps=100_000, tags={"symbol": "AAPL"}),
    ExperimentJob(PPO, env_config_msft, total_timesteps=100_000, tags={"symbol": "MSFT"}),
]

runner = BacktestRunner(verbose=True)
results = runner.run_batch(jobs)

BacktestRunner.inspect_batch(results)
```

`inspect_batch` renders a summary table across all jobs:

```
╔══════════════════════════════════════════════════════════════╗
║                   BATCH EXPERIMENT SUMMARY                   ║
╠════════╦══════╦══════════╦══════════╦══════════╦═══════════╣
║ ID     ║ Algo ║ Env      ║ Status   ║ Train %  ║ Test %    ║
╠════════╬══════╬══════════╬══════════╬══════════╬═══════════╣
║ a3f1b2 ║ PPO  ║ AAPL_... ║ ✓        ║ +12.34%  ║ +7.89%   ║
║ c8d4e5 ║ SAC  ║ AAPL_... ║ ✓        ║  +9.11%  ║ +5.22%   ║
║ f7a2b1 ║ PPO  ║ MSFT_... ║ ✓        ║ +15.02%  ║ +3.44%   ║
╚════════╩══════╩══════════╩══════════╩══════════╩═══════════╝
```

### Grid search with `JobGenerator`

`JobGenerator.generate_grid` creates the full combinatorial product of algorithms × env configs × algorithm configs:

```python
from quantrl_lab.experiments.backtesting.core import JobGenerator

# Multiple env configs (e.g. different reward functions or symbols)
env_configs = {
    "AAPL_sortino":  env_config_aapl_sortino,
    "AAPL_sharpe":   env_config_aapl_sharpe,
    "MSFT_sortino":  env_config_msft_sortino,
}

# Multiple hyperparameter configs to compare
algo_configs = [
    {"learning_rate": 3e-4, "n_steps": 2048},
    {"learning_rate": 1e-4, "n_steps": 4096},
]

jobs = JobGenerator.generate_grid(
    algorithms=[PPO, A2C],
    env_configs=env_configs,
    algorithm_configs=algo_configs,
    total_timesteps=100_000,
    num_eval_episodes=5,
)
# Produces 2 algos × 3 envs × 2 configs = 12 jobs

results = runner.run_batch(jobs)
BacktestRunner.inspect_batch(results)
```

---

## Metrics Reference

`MetricsCalculator` computes the following metrics from evaluation episodes. All metrics are prefixed with `train_` or `test_` in the result object.

| Metric key | Description |
|---|---|
| `avg_return_pct` | Mean portfolio return % across episodes |
| `std_return_pct` | Std-dev of return % across episodes |
| `win_rate` | Fraction of episodes with positive return |
| `avg_reward` | Mean total RL reward per episode |
| `std_reward` | Std-dev of RL reward |
| `avg_episode_length` | Mean steps per episode |
| `avg_sharpe_ratio` | Annualised Sharpe (252-day, 2% risk-free) averaged per episode |
| `avg_sortino_ratio` | Annualised Sortino (downside only) averaged per episode |
| `avg_max_drawdown` | Mean maximum drawdown across episodes (negative value) |
| `total_episodes` | Number of valid (non-errored) episodes |

!!! note
    Sharpe, Sortino, and drawdown require `detailed_actions` in episode data (the evaluation loop logs portfolio value at each step). These metrics are `None` if only summary episode data is available.

---

## Feature Importance: `AgentExplainer`

`AgentExplainer` runs the trained agent through one full episode and measures which observation features drive its decisions.

Two methods are available:

| Method | How it works | When used |
|---|---|---|
| `saliency` | Input × Gradient — computes the gradient of the policy output w.r.t. each observation feature | Default; requires PyTorch model (PPO, SAC, A2C) |
| `correlation` | Pearson correlation between each feature and the primary action value | Automatic fallback if saliency fails; also used for RecurrentPPO |

```python
from quantrl_lab.experiments.backtesting.explainer import AgentExplainer

explainer = AgentExplainer(model=result.model, env=test_env)

# Get top-10 most important features
top_features = explainer.analyze_feature_importance(top_k=10, method="saliency")

for feature, score in top_features.items():
    print(f"  {feature:40s}  {score:+.4f}")

# {
#   'RSI_14_t-0':              +0.4211
#   'portfolio_balance_ratio': -0.3102
#   'close_t-0':               +0.2894
#   'recent_volatility':       -0.2211
#   ...
# }
```

Feature importance is automatically computed inside `runner.run_job()` and stored in `result.top_features`. You only need to use `AgentExplainer` directly for custom post-hoc analysis.

---

## Hyperparameter Tuning: `OptunaRunner`

`OptunaRunner` wraps `BacktestRunner` with an Optuna study. Results are persisted to SQLite so runs can be resumed or analysed later.

### Defining a search space

The search space is a dict of param name → type + bounds:

```python
search_space = {
    "learning_rate": {"type": "float",       "low": 1e-5,  "high": 1e-2, "log": True},
    "n_steps":       {"type": "categorical",  "choices": [512, 1024, 2048, 4096]},
    "batch_size":    {"type": "categorical",  "choices": [64, 128, 256]},
    "gamma":         {"type": "float",        "low": 0.90,  "high": 0.999},
    "ent_coef":      {"type": "float",        "low": 1e-8,  "high": 0.1, "log": True},
}
```

Supported types: `"float"`, `"int"`, `"categorical"`, `"discrete_uniform"`.

**Pre-built search spaces** for common algorithms:

```python
from quantrl_lab.experiments.tuning.optuna_runner import (
    create_ppo_search_space,
    create_sac_search_space,
    create_a2c_search_space,
)

search_space = create_ppo_search_space()
```

### Running a study

```python
from quantrl_lab.experiments.backtesting.runner import BacktestRunner
from quantrl_lab.experiments.tuning.optuna_runner import OptunaRunner
from stable_baselines3 import PPO

runner = BacktestRunner(verbose=False)   # quiet during tuning
optuna_runner = OptunaRunner(
    runner=runner,
    storage_url="sqlite:///my_study.db",   # persists across runs
)

study = optuna_runner.optimize_hyperparameters(
    algo_class=PPO,
    env_config=env_config,
    search_space=create_ppo_search_space(),
    study_name="ppo_aapl_sortino",
    n_trials=50,
    total_timesteps=50_000,
    num_eval_episodes=3,
    optimization_metric="test_avg_return_pct",   # or "test_avg_sharpe_ratio"
    direction="maximize",
)

print(f"Best value:  {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Resuming a study

If the process is interrupted, re-running with the same `study_name` and `storage_url` continues from where it stopped (`load_if_exists=True` is set automatically):

```python
study = optuna_runner.optimize_hyperparameters(
    ...,
    study_name="ppo_aapl_sortino",   # same name → resumes
    n_trials=50,
)
```

### Querying results

```python
import optuna

study = optuna.load_study(
    study_name="ppo_aapl_sortino",
    storage="sqlite:///my_study.db",
)

# Best trial
print(study.best_trial.params)
print(study.best_trial.value)

# All trials as DataFrame
df = study.trials_dataframe()
print(df.sort_values("value", ascending=False).head(10))
```

---

## Full End-to-End Example

A complete workflow: fetch data → process → build env → run grid → tune best algo.

```python
from stable_baselines3 import PPO, SAC

from quantrl_lab.data.sources import YFinanceDataLoader
from quantrl_lab.data.processing.processor import DataProcessor
from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder
from quantrl_lab.experiments.backtesting.core import ExperimentJob, JobGenerator
from quantrl_lab.experiments.backtesting.runner import BacktestRunner
from quantrl_lab.experiments.tuning.optuna_runner import OptunaRunner, create_ppo_search_space
from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty

# ── 1. Data ──────────────────────────────────────────────────────────────────
loader = YFinanceDataLoader()
raw_df = loader.get_historical_ohlcv_data(["AAPL"], start="2018-01-01", end="2024-01-01")

processor = DataProcessor(ohlcv_data=raw_df)
splits, meta = processor.data_processing_pipeline(
    indicators=["RSI", {"SMA": {"window": 50}}, {"EMA": {"window": 20}}, "ATR", "MACD"],
    split_config={"train": 0.8, "test": 0.2},
)
train_df, test_df = splits["train"], splits["test"]

# ── 2. Build env configs ───────────────────────────────────────────────────────
def make_env_config(reward_strategy, name):
    return (
        BacktestEnvironmentBuilder()
        .with_data(train_data=train_df, test_data=test_df)
        .with_strategies(
            action=StandardActionStrategy(),
            reward=reward_strategy,
            observation=FeatureAwareObservationStrategy(),
        )
        .with_env_params(initial_balance=100_000, window_size=20)
        .build()
    )

env_configs = {
    "sortino":   make_env_config(DifferentialSortinoReward(), "sortino"),
    "composite": make_env_config(
        CompositeReward(
            strategies=[DifferentialSortinoReward(), DrawdownPenaltyReward(), InvalidActionPenalty()],
            weights=[0.6, 0.3, 0.1],
        ),
        "composite",
    ),
}

# ── 3. Grid search: PPO vs SAC on both reward configs ─────────────────────────
jobs = JobGenerator.generate_grid(
    algorithms=[PPO, SAC],
    env_configs=env_configs,
    total_timesteps=100_000,
    num_eval_episodes=5,
)

runner = BacktestRunner(verbose=True)
results = runner.run_batch(jobs)
BacktestRunner.inspect_batch(results)

# ── 4. Pick best combo, tune hyperparameters ──────────────────────────────────
# Assume PPO + composite reward won — tune it
best_env_config = env_configs["composite"]

optuna_runner = OptunaRunner(runner=BacktestRunner(verbose=False))
study = optuna_runner.optimize_hyperparameters(
    algo_class=PPO,
    env_config=best_env_config,
    search_space=create_ppo_search_space(),
    study_name="ppo_aapl_composite_v1",
    n_trials=30,
    total_timesteps=100_000,
    optimization_metric="test_avg_sharpe_ratio",
)

# ── 5. Re-train with best hyperparameters ──────────────────────────────────────
final_job = ExperimentJob(
    algorithm_class=PPO,
    env_config=best_env_config,
    algorithm_config=study.best_params,
    total_timesteps=200_000,
    num_eval_episodes=10,
    tags={"phase": "final", "tuned": "true"},
)

final_result = runner.run_job(final_job)
BacktestRunner.inspect_result(final_result)

# ── 6. Save model ─────────────────────────────────────────────────────────────
final_result.model.save("models/ppo_aapl_final")
```

---

## Integrating Alpha Signals

If you have run alpha research (see alpha research docs), you can inject validated signals directly into the environment's data pipeline via the builder:

```python
from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

# alpha_results: List[AlphaResult] from AlphaRunner
env_config = (
    BacktestEnvironmentBuilder()
    .with_data(train_data=train_df, test_data=test_df)
    .with_strategies(
        action=StandardActionStrategy(),
        reward=DifferentialSortinoReward(),
        observation=FeatureAwareObservationStrategy(),
    )
    .with_alpha_signals(
        results=alpha_results,
        top_n=5,               # use the 5 best signals by IC
        metric="ic",           # rank by information coefficient
    )
    .build()
)
```

The builder runs the top-N signals through `TechnicalIndicatorStep` before creating the env factories, so the processed DataFrames already include the alpha features when training begins.

You can also inject arbitrary pipeline steps for custom transformations:

```python
from quantrl_lab.data.processing.steps import MarketContextStep

env_config = (
    BacktestEnvironmentBuilder()
    .with_data(train_data=train_df, test_data=test_df)
    .with_strategies(...)
    .add_pipeline_step(MarketContextStep(sector_perf_df=sector_df))
    .build()
)
```

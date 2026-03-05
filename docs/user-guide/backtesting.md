# Backtesting

QuantRL-Lab uses a **job-based** backtesting architecture. An `ExperimentJob` describes a single training + evaluation run. `BacktestRunner` executes jobs and returns `ExperimentResult` objects.

For a complete walkthrough of a single job, see [Quickstart — Use BacktestRunner](../getting-started/quickstart.md#use-backtestrrunner-for-experiments). For the full API including `BacktestEnvironmentBuilder`, grid search, feature importance, and Optuna tuning, see [Experiments](../experiments.md).

## Running a Batch

Run multiple algorithm/config combinations and compare:

```python
from stable_baselines3 import PPO, SAC, A2C

jobs = [
    ExperimentJob(algorithm_class=PPO,  env_config=env_config, total_timesteps=100000),
    ExperimentJob(algorithm_class=SAC,  env_config=env_config, total_timesteps=100000),
    ExperimentJob(algorithm_class=A2C,  env_config=env_config, total_timesteps=100000),
]

results = runner.run_batch(jobs)
BacktestRunner.inspect_batch(results)
```

## Result Structure

`ExperimentResult` contains:

| Field | Description |
|-------|-------------|
| `result.status` | `"completed"` or `"failed"` |
| `result.metrics` | Dict with `train_avg_return_pct`, `test_avg_return_pct`, `test_avg_sharpe_ratio`, `test_avg_max_drawdown`, etc. |
| `result.model` | Trained stable-baselines3 model |
| `result.train_episodes` | List of per-episode dicts for training set |
| `result.test_episodes` | List of per-episode dicts for test set |
| `result.top_features` | Top correlated features from `AgentExplainer` |
| `result.execution_time` | Wall-clock seconds |

## Train/Test Split

Use ratio-based or date-based partitioning from `quantrl_lab.data.partitioning`:

```python
# Ratio-based (simple)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

# Date-based
from quantrl_lab.data.partitioning.date_range import DateRangeSplitter
splits = DateRangeSplitter({
    "train": ("2020-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2023-12-31"),
}).split(df)
train_df = splits["train"]
test_df = splits["test"]
```

## Saving and Loading Models

```python
# Save
result.model.save("ppo_aapl")

# Load and evaluate manually
from stable_baselines3 import PPO
model = PPO.load("ppo_aapl")

env = env_config.test_env_factory()
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
```

## See Also

- [Configuration](../getting-started/configuration.md) — environment and strategy parameters
- [Custom Strategies](custom-strategies.md) — build your own strategies
- `notebooks/backtesting_example.ipynb` — interactive end-to-end example

# Configuration

## Environment Configuration

QuantRL-Lab environments accept a configuration object that controls trading behavior, risk parameters, and episode settings.

## StockTradingConfig

```python
from quantrl_lab.environments.stock.stock_config import StockTradingConfig

config = StockTradingConfig(
    initial_balance=10000,        # Starting capital (USD)
    transaction_cost_pct=0.001,   # 0.1% per trade
    window_size=20,               # Lookback period for observations
    max_shares_per_trade=100,     # Position sizing limit
    enable_shorting=False,        # Allow short positions
    max_position_size=0.5         # Max 50% of portfolio in single position
)
```

## Configuration Parameters

### Capital & Risk

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_balance` | float | 10000 | Starting capital in USD |
| `transaction_cost_pct` | float | 0.001 | Transaction cost as % of trade value |
| `max_position_size` | float | 1.0 | Maximum position size as fraction of portfolio |

### Trading Rules

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_shorting` | bool | False | Allow short positions |
| `max_shares_per_trade` | int | None | Maximum shares per single trade (None = unlimited) |
| `min_shares_per_trade` | int | 1 | Minimum shares per trade |

### Episode Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 20 | Number of historical steps in observations |
| `max_episode_steps` | int | None | Maximum steps before termination (None = full dataset) |

## Strategy Configuration

=== "Action Strategy"

    ```python
    from quantrl_lab.environments.stock.strategies.actions import StandardMarketActionStrategy

    action_strategy = StandardMarketActionStrategy(
        max_shares_per_trade=100,
        allow_partial_fills=True
    )
    ```

=== "Observation Strategy"

    ```python
    from quantrl_lab.environments.stock.strategies.observations import PortfolioWithTrendObservation

    observation_strategy = PortfolioWithTrendObservation(
        include_indicators=['SMA_20', 'RSI_14', 'MACD'],
        normalize=True
    )
    ```

=== "Reward Strategy"

    ```python
    from quantrl_lab.environments.stock.strategies.rewards import WeightedCompositeReward

    # Use preset
    reward_strategy = WeightedCompositeReward.from_preset("balanced")

    # Or custom weights
    reward_strategy = WeightedCompositeReward(
        components=[
            PortfolioValueChangeReward(),
            TrendFollowingReward(),
            InvalidActionPenalty()
        ],
        weights=[0.5, 0.3, 0.2]  # Must sum to 1.0
    )
    ```

## Reward Presets

Available presets in `WeightedCompositeReward.from_preset()`:

| Preset | Description |
|--------|-------------|
| `conservative` | Emphasizes portfolio value growth and penalizes risk |
| `explorative` | Encourages trading and exploration |
| `balanced` | Balanced approach between growth and risk management |
| `risk_managed` | Strong focus on downside protection |

!!! tip "Choosing a preset"
    Start with `balanced` for general-purpose training. Use `conservative` if
    your agent takes too many risky trades, or `explorative` if it learns to
    hold indefinitely.

See [`reward_presets.py`](https://github.com/whanyu1212/QuantRL-Lab/blob/main/src/quantrl_lab/experiments/backtesting/config/reward_presets.py) for preset definitions.

## Data Configuration

### Data Sources

=== "YFinance (free)"

    ```python
    from quantrl_lab.data.sources import YFinanceDataLoader

    loader = YFinanceDataLoader()
    df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start="2020-01-01",
        end="2023-12-31",
        timeframe="1d"  # (1)!
    )
    ```

    1. Supported intervals: `1m`, `5m`, `1h`, `1d`, `1wk`, `1mo`

=== "Alpaca (requires API key)"

    ```python
    from quantrl_lab.data.sources import AlpacaDataLoader
    import os

    loader = AlpacaDataLoader(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )
    df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start="2020-01-01",
        end="2023-12-31",
        timeframe="1d"
    )
    ```

### Technical Indicators

```python
from quantrl_lab.data.processors.processor import DataProcessor
from quantrl_lab.data.indicators import IndicatorRegistry

# List available indicators
IndicatorRegistry.list_all()
# Output: ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', ...]

# Apply indicators
processor = DataProcessor()
df = processor.apply_indicators(
    df,
    indicators=["SMA", "EMA", "RSI", "MACD"],
    sma_window=20,   # (1)!
    ema_window=12,
    rsi_window=14
)
```

1. Each indicator accepts its own keyword arguments for customization

## Complete Example

??? example "Full configuration example (click to expand)"

    ```python
    from quantrl_lab.data.sources.yfinance import YFinanceDataLoader
    from quantrl_lab.data.processors.processor import DataProcessor
    from quantrl_lab.environments.stock.env_single_stock import SingleStockTradingEnv
    from quantrl_lab.environments.stock.stock_config import StockTradingConfig
    from quantrl_lab.environments.stock.strategies import (
        StandardMarketActionStrategy,
        PortfolioWithTrendObservation,
        WeightedCompositeReward
    )

    # 1. Load data
    loader = YFinanceDataLoader()
    df = loader.get_historical_ohlcv_data(symbols="AAPL", start="2020-01-01", end="2023-12-31")

    # 2. Add indicators
    processor = DataProcessor()
    df = processor.apply_indicators(df, indicators=["SMA", "EMA", "RSI"])

    # 3. Configure environment
    config = StockTradingConfig(
        initial_balance=10000,
        transaction_cost_pct=0.001,
        window_size=20,
        enable_shorting=False
    )

    # 4. Define strategies
    action_strategy = StandardMarketActionStrategy()
    observation_strategy = PortfolioWithTrendObservation()
    reward_strategy = WeightedCompositeReward.from_preset("balanced")

    # 5. Create environment
    env = SingleStockTradingEnv(
        data=df,
        config=config,
        action_strategy=action_strategy,
        observation_strategy=observation_strategy,
        reward_strategy=reward_strategy
    )
    ```

## Next Steps

- [Quickstart](quickstart.md) - Train your first agent
- [Custom Strategies](../user-guide/custom-strategies.md) - Build custom strategies
- [Backtesting](../user-guide/backtesting.md) - Advanced workflows

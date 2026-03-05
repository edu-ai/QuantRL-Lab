# Configuration

## Environment Configuration

QuantRL-Lab environments accept a configuration object that controls trading behavior, risk parameters, and episode settings.

## SingleStockEnvConfig

```python
from quantrl_lab.environments.stock.components.config import SingleStockEnvConfig, SimulationConfig

config = SingleStockEnvConfig(
    initial_balance=100000.0,     # Starting capital (USD)
    window_size=20,               # Lookback period for observations
    simulation=SimulationConfig(
        transaction_cost_pct=0.001,      # 0.1% per trade
        slippage=0.001,                  # Slippage percentage
        enable_shorting=False,           # Allow short positions
        order_expiration_steps=5,        # Steps before pending order expires
        ignore_fees=False,               # Whether to ignore transaction costs
    )
)
```

## Configuration Parameters

### `SingleStockEnvConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_balance` | float | 100000.0 | Starting capital in USD |
| `window_size` | int | 20 | Number of historical steps in observations |
| `price_column_index` | int | 0 | Index of price column in data array |

### `SimulationConfig` (nested under `simulation`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transaction_cost_pct` | float | 0.001 | Transaction cost as % of trade value |
| `slippage` | float | 0.001 | Slippage percentage for market orders |
| `enable_shorting` | bool | False | Allow short positions |
| `order_expiration_steps` | int | 5 | Steps before a pending order expires |
| `ignore_fees` | bool | False | Whether to ignore transaction costs |

### `RewardConfig` (nested under `rewards`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_range` | tuple | (-1.0, 1.0) | Range to clip the final reward |

For strategy configuration details (action space, observation features, reward components), see [Environments](../environments.md).

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

For the full list of available indicators and pipeline steps, see [Data Processing](../data-processing.md). Quick example:

```python
from quantrl_lab.data.processing.processor import DataProcessor

processor = DataProcessor(ohlcv_data=df)
df, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "RSI", {"EMA": {"window": 20}}, "MACD"]
)
```

## Next Steps

- [Quickstart](quickstart.md) - Train your first agent
- [Custom Strategies](../user-guide/custom-strategies.md) - Build custom strategies
- [Backtesting](../user-guide/backtesting.md) - Advanced workflows

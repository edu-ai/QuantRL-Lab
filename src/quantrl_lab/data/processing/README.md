# Data Processing Pipeline

This module implements a composable, builder-pattern data processing pipeline for financial time-series data. It is designed to transform raw OHLCV (Open-High-Low-Close-Volume) data into feature-rich datasets ready for reinforcement learning or other analysis.

## Architecture

The system is built around three core concepts:

1.  **DataPipeline**: A container that manages a sequence of processing steps.
2.  **ProcessingStep**: A protocol defining a single transformation unit.
3.  **DataProcessor**: A high-level facade that coordinates data loading, configuration, and pipeline execution.

### Components

- **`pipeline.py`**: Contains `DataPipeline`, responsible for chaining steps and executing them sequentially.
- **`processor.py`**: Contains `DataProcessor`, the main entry point for most use cases. It handles data sources, splitting (train/test), and configuration.
- **`steps/`**: Directory containing concrete implementations of processing steps.
- **`features/`**: Logic for specific feature generation (e.g., technical indicators, sentiment) used by steps.

## Usage

### High-Level API (Recommended)

The `DataProcessor` class provides a unified interface for standard workflows:

```python
from quantrl_lab.data.processing.processor import DataProcessor

# Initialize with raw data
processor = DataProcessor(ohlcv_data=df, news_data=news_df)

# Run standard pipeline
processed_data, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "RSI"],
    split_config={'train': 0.7, 'test': 0.3}
)
```

### Low-Level Builder API (Custom Workflows)

For more control, you can construct a `DataPipeline` manually:

```python
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import (
    TechnicalIndicatorStep,
    ColumnCleanupStep
)

# Build pipeline
pipeline = (DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=["SMA", {"RSI": {"window": 14}}]))
    .add_step(ColumnCleanupStep(columns_to_drop=["Date"]))
)

# Execute
result_df, metadata = pipeline.execute(raw_df)
```

### Integration with DataSourceRegistry

You can easily integrate with the `DataSourceRegistry` to fetch data from configured sources (Alpaca, YFinance, etc.) and pass it to the processor:

```python
from quantrl_lab.data.source_registry import DataSourceRegistry
from quantrl_lab.data.processing.processor import DataProcessor

# 1. Get data from registry
registry = DataSourceRegistry()

# Fetch OHLCV data (Primary Source)
ohlcv_df = registry.get_historical_ohlcv_data(
    symbols="AAPL",
    start="2023-01-01",
    end="2023-12-31",
    timeframe="1d"
)

# Fetch News data (Optional)
news_df = registry.get_news_data(
    symbols="AAPL",
    start="2023-01-01",
    end="2023-12-31"
)

# 2. Initialize Processor
processor = DataProcessor(
    ohlcv_data=ohlcv_df,
    news_data=news_df
)

# 3. Run Pipeline
processed_data, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "RSI"],
    split_config={'train': 0.7, 'test': 0.3}
)
```

## Available Steps

| Step Class | Description |
|------------|-------------|
| `TechnicalIndicatorStep` | Adds technical indicators (SMA, RSI, MACD, etc.) using `TechnicalFeatureGenerator`. |
| `SentimentEnrichmentStep` | Merges news data and computes sentiment scores. |
| `NumericConversionStep` | Converts specified columns to numeric types, handling errors. |
| `ColumnCleanupStep` | Drops unwanted columns (e.g., raw dates, symbols) to prepare for model input. |

## Metadata Tracking

The pipeline automatically tracks metadata through `ProcessingMetadata`, which records:
- Original and final data shapes
- Applied technical indicators
- Dropped columns
- Date ranges for splits

This ensures reproducibility and allows the environment to know how the data was transformed.

## Extending the Pipeline

To create a custom processing step, implement the `ProcessingStep` protocol defined in `steps/base.py`:

```python
from quantrl_lab.data.processing.processor import ProcessingMetadata
import pandas as pd

class MyCustomStep:
    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        # Perform transformation
        data["new_col"] = data["close"] * 2
        return data

    def get_step_name(self) -> str:
        return "My Custom Step"
```

Then add it to your pipeline:
```python
pipeline.add_step(MyCustomStep())
```

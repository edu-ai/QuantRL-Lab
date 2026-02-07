# Data Processing

This section covers the data processing pipeline, which transforms raw OHLCV data into feature-rich datasets ready for reinforcement learning.

## Core Components

The processing pipeline is built around three main components:

1. **DataProcessor**: High-level facade for common operations
2. **DataPipeline**: Builder-pattern pipeline for composing transformations
3. **ProcessingStep**: Individual transformation units (indicators, sentiment, etc.)

## DataProcessor

The `DataProcessor` provides a high-level interface for applying transformations.

::: quantrl_lab.data.processing.processor.DataProcessor
    options:
        members:
            - __init__
            - data_processing_pipeline
            - load_indicators
            - append_technical_indicators
            - append_news_sentiment_data
            - drop_unwanted_columns
            - convert_columns_to_numeric

::: quantrl_lab.data.processing.processor.ProcessingMetadata
    options:
        members:
            - to_dict

## DataPipeline (Builder Pattern)

For more control, use the `DataPipeline` class directly to chain processing steps.

::: quantrl_lab.data.processing.pipeline.DataPipeline
    options:
        members:
            - add_step
            - execute
            - get_steps

### Example Usage

```python
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import (
    TechnicalIndicatorStep,
    SentimentEnrichmentStep,
    ColumnCleanupStep
)

# Build a custom pipeline
pipeline = (DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=["SMA", "RSI"]))
    .add_step(SentimentEnrichmentStep(news_data=news_df))
    .add_step(ColumnCleanupStep(columns_to_drop=["Date"]))
)

# Execute
processed_df, metadata = pipeline.execute(raw_df)
```

## Processing Steps

Individual steps that can be added to a pipeline.

### Base Interface

::: quantrl_lab.data.processing.steps.base.ProcessingStep

### Available Steps

::: quantrl_lab.data.processing.steps.technical.TechnicalIndicatorStep
::: quantrl_lab.data.processing.steps.sentiment.SentimentEnrichmentStep
::: quantrl_lab.data.processing.steps.analyst.AnalystEstimatesStep
::: quantrl_lab.data.processing.steps.context.MarketContextStep
::: quantrl_lab.data.processing.steps.conversion.NumericConversionStep
::: quantrl_lab.data.processing.steps.cleanup.ColumnCleanupStep

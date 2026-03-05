# Data Processing API

::: quantrl_lab.data.processing.processor.DataProcessor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Data Pipeline

The `DataPipeline` allows for flexible, composable data transformations.

::: quantrl_lab.data.processing.pipeline.DataPipeline

## Processing Steps

Processing steps encapsulate individual transformations. They can be chained
together in a pipeline.

### Base Interface

::: quantrl_lab.data.processing.steps.base.ProcessingStep

### Available Steps

::: quantrl_lab.data.processing.steps.features.technical.TechnicalIndicatorStep
::: quantrl_lab.data.processing.steps.alternative.sentiment.SentimentEnrichmentStep
::: quantrl_lab.data.processing.steps.alternative.analyst.AnalystEstimatesStep
::: quantrl_lab.data.processing.steps.features.context.MarketContextStep
::: quantrl_lab.data.processing.steps.cleaning.conversion.NumericConversionStep
::: quantrl_lab.data.processing.steps.cleaning.cleanup.ColumnCleanupStep

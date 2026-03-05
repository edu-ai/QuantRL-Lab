# Feature Engineering Example

## Sentiment Analysis with Exponential Decay

This example demonstrates how to use the `exponential_decay` strategy for sentiment analysis. This strategy is superior to forward-filling because it models the diminishing impact of news over time.

### Configuration

The `SentimentFeatureGenerator` now accepts a `decay_rate` parameter.

- `fillna_strategy="exponential_decay"`: Activates the decay logic.
- `decay_rate` (float): The factor by which the sentiment score is multiplied each day without new news. Default is `0.8`.

### Example Usage

```python
from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator
from quantrl_lab.data.processing.sentiment import HuggingFaceProvider, SentimentConfig

# Initialize provider and config
provider = HuggingFaceProvider()
config = SentimentConfig(text_column="headline", date_column="date")

# Create generator with exponential decay
# A decay rate of 0.8 means the signal retains 80% of its strength each day.
# Day 0: 1.0 (News event)
# Day 1: 0.8
# Day 2: 0.64
# ...
generator = SentimentFeatureGenerator(
    sentiment_provider=provider,
    sentiment_config=config,
    news_data=news_df,
    fillna_strategy="exponential_decay",
    decay_rate=0.8
)

# Generate features
enriched_df = generator.generate(ohlcv_df)
```

## Analyst Estimates Integration

The pipeline also supports integrating analyst grades (e.g., "Strong Buy", "Hold").

### Monthly Alignment

Analyst grades are often released on a monthly basis or irregularly. The pipeline now automatically aligns these monthly grades with daily price data by matching the **Year-Month**. This ensures that a grade released on a weekend or holiday (e.g., Jan 1st) is correctly applied to the first trading day of the month (e.g., Jan 3rd).

```python
from quantrl_lab.data.processing import DataProcessor

# The processor automatically handles the alignment
processor = DataProcessor(
    ohlcv_data=ohlcv_df,
    analyst_grades=grades_df
)

processed_data, _ = processor.data_processing_pipeline()
```

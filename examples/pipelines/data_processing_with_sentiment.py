"""
Data Processing Pipeline with Sentiment Analysis.

This example demonstrates:
- Loading both OHLCV and news data
- Applying sentiment analysis to news
- Merging sentiment with price data
- Custom sentiment providers
- Different fillna strategies

Requires: ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file
"""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

from quantrl_lab.data import AlpacaDataLoader, DataProcessor
from quantrl_lab.data.processing.sentiment import HuggingFaceConfig, HuggingFaceProvider, SentimentConfig


def check_api_keys():
    """Check if required API keys are configured."""
    load_dotenv()

    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        print("❌ Error: Alpaca API keys not found!")
        print("\nTo run this example:")
        print("1. Copy .env.example to .env")
        print("2. Add your Alpaca API keys to .env:")
        print("   ALPACA_API_KEY=your_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_here")
        print("\nGet free API keys at: https://alpaca.markets/")
        return False
    return True


def main():
    print("=" * 80)
    print("Data Processing with Sentiment Analysis Example")
    print("=" * 80)

    if not check_api_keys():
        return None, None

    # Step 1: Load OHLCV and news data
    print("\n[1/4] Loading OHLCV and news data from Alpaca...")
    loader = AlpacaDataLoader()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days

    # Load price data
    ohlcv_df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timeframe="1d",
    )

    # Load news data
    news_df = loader.get_news_data(
        symbols="AAPL",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )

    print(f"✓ Loaded {len(ohlcv_df)} price rows")
    print(f"✓ Loaded {len(news_df)} news articles")
    print(f"  News columns: {list(news_df.columns)}")

    # Step 2: Configure sentiment analysis
    print("\n[2/4] Configuring sentiment analysis...")

    # Option A: Use default HuggingFace provider (FinBERT)
    print("  Using FinBERT model for financial sentiment analysis")

    # Create custom sentiment config
    sentiment_config = SentimentConfig(
        text_column="headline",  # Column containing text to analyze
        date_column="created_at",  # Column containing date
        sentiment_score_column="sentiment_score",  # Output column name
    )

    # Create HuggingFace provider with custom settings
    hf_config = HuggingFaceConfig(
        model_name="ProsusAI/finbert",  # Financial sentiment model
        device=-1,  # Use CPU (-1) or GPU (0, 1, etc.)
        max_length=512,  # Maximum token length
        truncation=True,  # Truncate long texts
    )
    sentiment_provider = HuggingFaceProvider(hf_config)

    # Step 3: Process with sentiment
    print("\n[3/4] Processing data with sentiment analysis...")

    processor = DataProcessor(
        ohlcv_data=ohlcv_df,
        news_data=news_df,
        sentiment_config=sentiment_config,
        sentiment_provider=sentiment_provider,
    )

    # Define technical indicators
    indicators = [
        {"SMA": {"window": 10}},
        {"SMA": {"window": 20}},
        "RSI",
        "MACD",
    ]

    # Process with neutral fillna strategy
    print("  Strategy: Fill missing sentiment with neutral (0.0)")
    processed_data, metadata = processor.data_processing_pipeline(
        indicators=indicators,
        fillna_strategy="neutral",  # Fill missing sentiment with 0.0
    )

    print("✓ Processing complete")
    print(f"  Sentiment applied: {metadata['news_sentiment_applied']}")
    print(f"  Final shape: {processed_data.shape}")

    # Step 4: Analyze sentiment impact
    print("\n[4/4] Analyzing sentiment data:")
    print("-" * 80)

    if "sentiment_score" in processed_data.columns:
        sentiment_stats = processed_data["sentiment_score"].describe()
        print("Sentiment Score Statistics:")
        print(sentiment_stats)

        # Count sentiment distribution
        positive = (processed_data["sentiment_score"] > 0.1).sum()
        neutral = ((processed_data["sentiment_score"] >= -0.1) & (processed_data["sentiment_score"] <= 0.1)).sum()
        negative = (processed_data["sentiment_score"] < -0.1).sum()

        print("\nSentiment Distribution:")
        print(f"  Positive (>0.1): {positive} days ({positive/len(processed_data)*100:.1f}%)")
        print(f"  Neutral (-0.1 to 0.1): {neutral} days ({neutral/len(processed_data)*100:.1f}%)")
        print(f"  Negative (<-0.1): {negative} days ({negative/len(processed_data)*100:.1f}%)")

        # Show correlation with price change
        if "Close" in processed_data.columns:
            # Calculate daily returns
            processed_data_copy = processed_data.copy()
            processed_data_copy["Returns"] = processed_data_copy["Close"].pct_change()

            # Correlation with returns
            correlation = processed_data_copy[["sentiment_score", "Returns"]].corr().iloc[0, 1]
            print(f"\nCorrelation between sentiment and next-day returns: {correlation:.4f}")

    print("\n" + "=" * 80)
    print("Comparison: Different Fillna Strategies")
    print("=" * 80)

    # Process with forward fill strategy
    print("\nProcessing with 'fill_forward' strategy...")
    processor2 = DataProcessor(
        ohlcv_data=ohlcv_df,
        news_data=news_df,
        sentiment_config=sentiment_config,
        sentiment_provider=sentiment_provider,
    )

    processed_data_ff, metadata_ff = processor2.data_processing_pipeline(
        indicators=indicators,
        fillna_strategy="fill_forward",  # Forward fill missing sentiment
    )

    print("Comparison of sentiment handling:")
    print(f"  Neutral strategy - zeros: {(processed_data['sentiment_score'] == 0.0).sum()}")
    print(f"  Forward fill strategy - zeros: {(processed_data_ff['sentiment_score'] == 0.0).sum()}")

    print("\n✓ Sentiment processing complete!")
    print("  Data includes both technical indicators and news sentiment")

    return processed_data, metadata


if __name__ == "__main__":
    data, metadata = main()

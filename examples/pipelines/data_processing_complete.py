"""
Complete Data Processing Pipeline Showcase.

This example demonstrates the full power of DataProcessor by combining:
- Multiple data sources (OHLCV + news)
- Advanced indicator configurations
- Sentiment analysis
- Date-based splitting
- Comprehensive metadata tracking

This is a comprehensive example showing all features together.

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
        print("2. Add your Alpaca API keys")
        print("\nGet free API keys at: https://alpaca.markets/")
        return False
    return True


def main():
    print("=" * 80)
    print("COMPLETE DATA PROCESSING PIPELINE SHOWCASE")
    print("=" * 80)
    print("\nThis example demonstrates all DataProcessor features:")
    print("  ✓ Technical indicators with custom parameters")
    print("  ✓ News sentiment analysis")
    print("  ✓ Date-based train/val/test splitting")
    print("  ✓ Comprehensive metadata tracking")
    print("  ✓ Data quality validation")

    if not check_api_keys():
        return None, None

    # Configuration
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)

    loader = AlpacaDataLoader()

    print(f"\nLoading OHLCV data for {symbol}...")
    ohlcv_df = loader.get_historical_ohlcv_data(
        symbols=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timeframe="1d",
    )
    print(f"✓ Loaded {len(ohlcv_df)} price rows")

    print(f"\nLoading news data for {symbol}...")
    news_df = loader.get_news_data(
        symbols=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    print(f"✓ Loaded {len(news_df)} news articles")

    # Step 2: Configure indicators
    print("\n" + "=" * 80)
    print("STEP 2: INDICATOR CONFIGURATION")
    print("=" * 80)

    indicators = [
        # Trend indicators - multiple timeframes using list-of-values syntax
        {"SMA": {"window": [10, 20, 50, 200]}},
        {"EMA": {"window": [12, 26]}},
        # Momentum indicators
        {"RSI": {"window": [14, 21]}},
        "MACD",
        "STOCH",
        # Volatility indicators - multiple combinations using product logic
        {"BB": {"window": [10, 20], "num_std": [1.5, 2.0]}},
        "ATR",
        # Volume indicators
        "OBV",
    ]

    print(f"\n✓ Configured {len(indicators)} indicator entries (generating multiple timeframes automatically)")
    print("  Categories:")
    print("    - Trend: SMA (4 windows), EMA (2 windows)")
    print("    - Momentum: RSI (2 windows), MACD, Stochastic")
    print("    - Volatility: Bollinger Bands (4 combinations), ATR")
    print("    - Volume: OBV")

    # Step 3: Configure sentiment
    print("\n" + "=" * 80)
    print("STEP 3: SENTIMENT ANALYSIS CONFIGURATION")
    print("=" * 80)

    sentiment_config = SentimentConfig(
        text_column="headline",
        date_column="created_at",
        sentiment_score_column="sentiment_score",
    )

    hf_config = HuggingFaceConfig(
        model_name="ProsusAI/finbert",
        device=-1,
        max_length=512,
        truncation=True,
    )

    sentiment_provider = HuggingFaceProvider(hf_config)

    print("✓ Sentiment analysis configured")
    print("  Provider: HuggingFace (FinBERT)")
    print(f"  Model: {hf_config.model_name}")
    print(f"  Text field: {sentiment_config.text_column}")

    # Step 4: Configure date-based splits
    print("\n" + "=" * 80)
    print("STEP 4: SPLIT CONFIGURATION")
    print("=" * 80)

    # Calculate split dates (60% train, 20% val, 20% test)
    total_days = (end_date - start_date).days
    train_end = start_date + timedelta(days=int(total_days * 0.6))
    val_end = start_date + timedelta(days=int(total_days * 0.8))

    split_config = {
        "train": (start_date.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        "val": (train_end.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
        "test": (val_end.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    }

    print("\n✓ Date-based split configured (60/20/20)")
    print(f"  Train: {split_config['train'][0]} to {split_config['train'][1]}")
    print(f"  Val:   {split_config['val'][0]} to {split_config['val'][1]}")
    print(f"  Test:  {split_config['test'][0]} to {split_config['test'][1]}")

    # Step 5: Process data
    print("\n" + "=" * 80)
    print("STEP 5: DATA PROCESSING")
    print("=" * 80)

    processor = DataProcessor(
        ohlcv_data=ohlcv_df,
        news_data=news_df,
        sentiment_config=sentiment_config,
        sentiment_provider=sentiment_provider,
    )

    print("\nProcessing pipeline running...")
    split_data, metadata = processor.data_processing_pipeline(
        indicators=indicators,
        fillna_strategy="fill_forward",
        split_config=split_config,
        verbose=True,
    )

    print("\n✓ Processing complete!")

    # Step 6: Results and analysis
    print("\n" + "=" * 80)
    print("STEP 6: RESULTS & ANALYSIS")
    print("=" * 80)

    # Data shapes
    print("\nData Shapes:")
    for split_name in ["train", "val", "test"]:
        shape = metadata["final_shapes"][split_name]
        print(f"  {split_name.capitalize():>5}: {shape[0]:4d} rows × {shape[1]:3d} features")

    # Metadata summary
    print("\nMetadata Summary:")
    print(f"  Symbol: {metadata['symbol']}")
    print(f"  Original shape: {metadata['original_shape']}")
    print(f"  Indicators applied: {len(metadata['technical_indicators'])}")
    print(f"  Sentiment analysis: {metadata['news_sentiment_applied']}")
    print(f"  Fillna strategy: {metadata['fillna_strategy']}")
    print(f"  Columns dropped: {metadata['columns_dropped']}")

    # Feature categories
    train_df = split_data["train"]
    all_cols = list(train_df.columns)

    sma_features = [c for c in all_cols if c.startswith("SMA_")]
    ema_features = [c for c in all_cols if c.startswith("EMA_")]
    rsi_features = [c for c in all_cols if c.startswith("RSI_")]
    bb_features = [c for c in all_cols if c.startswith("BB_")]
    macd_features = [c for c in all_cols if c.startswith("MACD")]
    other_indicators = [
        c
        for c in all_cols
        if c not in ["Open", "High", "Low", "Close", "Volume", "sentiment_score"]
        and not any(c.startswith(prefix) for prefix in ["SMA_", "EMA_", "RSI_", "BB_", "MACD"])
    ]

    print("\nFeature Categories:")
    print("  OHLCV: 5 features")
    print("  Sentiment: 1 feature (sentiment_score)")
    print(f"  SMA: {len(sma_features)} features - {sma_features}")
    print(f"  EMA: {len(ema_features)} features - {ema_features}")
    print(f"  RSI: {len(rsi_features)} features - {rsi_features}")
    print(f"  Bollinger Bands: {len(bb_features)} features")
    print(f"  MACD: {len(macd_features)} features - {macd_features}")
    print(f"  Other indicators: {len(other_indicators)} features - {other_indicators}")

    # Sentiment analysis
    if "sentiment_score" in train_df.columns:
        print("\nSentiment Analysis:")
        all_sentiment = [split_data[s]["sentiment_score"] for s in ["train", "val", "test"]]
        import pandas as pd

        combined_sentiment = pd.concat(all_sentiment)

        print(f"  Mean: {combined_sentiment.mean():.4f}")
        print(f"  Std: {combined_sentiment.std():.4f}")
        print(f"  Min: {combined_sentiment.min():.4f}")
        print(f"  Max: {combined_sentiment.max():.4f}")

        positive = (combined_sentiment > 0.1).sum()
        neutral = ((combined_sentiment >= -0.1) & (combined_sentiment <= 0.1)).sum()
        negative = (combined_sentiment < -0.1).sum()
        total = len(combined_sentiment)

        print("\n  Distribution:")
        print(f"    Positive: {positive}/{total} ({positive/total*100:.1f}%)")
        print(f"    Neutral:  {neutral}/{total} ({neutral/total*100:.1f}%)")
        print(f"    Negative: {negative}/{total} ({negative/total*100:.1f}%)")

    # Data quality
    print("\nData Quality:")
    for split_name in ["train", "val", "test"]:
        df = split_data[split_name]
        missing = df.isna().sum().sum()
        infinite = df.isin([float('inf'), float('-inf')]).sum().sum()
        print(f"  {split_name.capitalize():>5}: Missing={missing:3d}, Infinite={infinite:3d}")

    # Feature correlations with Close
    print("\nTop 10 Features Correlated with Close (Train Set):")
    correlations = train_df.corr()["Close"].sort_values(ascending=False)
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature:20s}: {corr:7.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully processed {symbol} data")
    print(f"✓ Created {len(split_data)} datasets with {train_df.shape[1]} features each")
    print(f"✓ Includes {len(metadata['technical_indicators'])} technical indicators")
    print(f"✓ Integrated news sentiment from {len(news_df)} articles")
    print("✓ Ready for machine learning model training!")
    print("\nDatasets available in 'split_data' dict:")
    print("  - split_data['train']")
    print("  - split_data['val']")
    print("  - split_data['test']")
    print("\nMetadata available in 'metadata' dict with full processing details")

    return split_data, metadata


if __name__ == "__main__":
    data, metadata = main()

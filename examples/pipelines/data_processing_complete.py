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

from quantrl_lab.data import DataProcessor
from quantrl_lab.data.processing.sentiment import HuggingFaceConfig, HuggingFaceProvider, SentimentConfig
from quantrl_lab.data.source_registry import DataSourceRegistry


def check_api_keys():
    """Check if required API keys are configured."""
    load_dotenv()

    missing = []
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        missing.append("Alpaca (ALPACA_API_KEY, ALPACA_SECRET_KEY)")

    if not os.getenv("FMP_API_KEY"):
        missing.append("Financial Modeling Prep (FMP_API_KEY)")

    if missing:
        print(f"❌ Error: Missing API keys: {', '.join(missing)}")
        print("\nTo run this example:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys")
        return False
    return True


def main():
    print("=" * 80)
    print("COMPLETE DATA PROCESSING PIPELINE SHOWCASE")
    print("=" * 80)
    print("\nThis example demonstrates all DataProcessor features:")
    print("  ✓ Technical indicators with custom parameters")
    print("  ✓ News sentiment analysis")
    print("  ✓ Analyst estimates (Grades & Ratings)")
    print("  ✓ Market context (Sector & Industry performance)")
    print("  ✓ Date-based train/val/test splitting")
    print("  ✓ Comprehensive metadata tracking")
    print("  ✓ Data quality validation")

    if not check_api_keys():
        return None, None

    # Configuration
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # ~13 months to cover 200-day SMA

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)

    # Initialize Registry
    registry = DataSourceRegistry()
    loader = registry.primary_source
    fmp_loader = registry.get_source("fundamental_source")

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

    print(f"\nLoading analyst & fundamental data for {symbol}...")
    # Get profile for sector/industry context
    profile = fmp_loader.get_company_profile(symbol)
    sector = profile.iloc[0].get('sector') if not profile.empty else None
    industry = profile.iloc[0].get('industry') if not profile.empty else None
    print(f"  Sector: {sector}, Industry: {industry}")

    # Fetch Analyst Data
    grades_df = fmp_loader.get_historical_grades(symbol)
    ratings_df = fmp_loader.get_historical_rating(symbol, limit=200)
    print(f"✓ Loaded {len(grades_df)} grades and {len(ratings_df)} ratings")

    # Fetch Market Context
    sector_perf_df = None
    industry_perf_df = None
    if sector:
        sector_perf_df = fmp_loader.get_historical_sector_performance(
            sector, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )
        print(f"✓ Loaded {len(sector_perf_df)} sector performance records")
    if industry:
        industry_perf_df = fmp_loader.get_historical_industry_performance(
            industry, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )
        print(f"✓ Loaded {len(industry_perf_df)} industry performance records")

    # Step 2: Configure indicators
    print("\n" + "=" * 80)
    print("STEP 2: INDICATOR CONFIGURATION")
    print("=" * 80)

    indicators = [
        # Trend indicators
        {"SMA": {"window": [10, 20, 50, 200]}},
        {"EMA": {"window": [12, 26]}},
        # Momentum indicators
        {"RSI": {"window": [14, 21]}},
        "MACD",
        "STOCH",
        # Volatility indicators
        {"BB": {"window": [10, 20], "num_std": [1.5, 2.0]}},
        "ATR",
        # Volume indicators
        "OBV",
    ]

    print(f"\n✓ Configured {len(indicators)} indicator entries")

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
        batch_size=32,
    )

    sentiment_provider = HuggingFaceProvider(hf_config)
    print("✓ Sentiment analysis configured (FinBERT)")

    # Step 4: Configure date-based splits
    print("\n" + "=" * 80)
    print("STEP 4: SPLIT CONFIGURATION")
    print("=" * 80)

    total_days = (end_date - start_date).days
    train_end = start_date + timedelta(days=int(total_days * 0.6))
    val_end = start_date + timedelta(days=int(total_days * 0.8))

    split_config = {
        "train": (start_date.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        "val": (train_end.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
        "test": (val_end.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    }

    print("\n✓ Date-based split configured (60/20/20)")

    # Step 5: Process data
    print("\n" + "=" * 80)
    print("STEP 5: DATA PROCESSING")
    print("=" * 80)

    processor = DataProcessor(
        ohlcv_data=ohlcv_df,
        news_data=news_df,
        analyst_grades=grades_df,
        analyst_ratings=ratings_df,
        sector_performance=sector_perf_df,
        industry_performance=industry_perf_df,
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
    print(f"  Indicators applied: {len(metadata['technical_indicators'])}")
    print(f"  Sentiment applied: {metadata['news_sentiment_applied']}")
    print(f"  Analyst data applied: {metadata.get('analyst_data_applied', False)}")
    print(f"  Market context applied: {metadata.get('market_context_applied', False)}")

    # Feature categories
    train_df = split_data["train"]
    all_cols = list(train_df.columns)

    sma_features = [c for c in all_cols if c.startswith("SMA_")]
    analyst_features = [c for c in all_cols if c.startswith("analyst_")]
    market_features = [c for c in all_cols if c.startswith("sector_") or c.startswith("industry_")]

    print("\nFeature Categories:")
    print("  OHLCV: 5 features")
    print("  Sentiment: 1 feature (sentiment_score)")
    print(f"  SMA: {len(sma_features)} features")
    print(f"  Analyst Estimates: {len(analyst_features)} features")
    print(f"  Market Context: {len(market_features)} features")
    print(f"  Total Features: {len(all_cols)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully processed {symbol} data")
    print("✓ Integrated Technicals, Sentiment, Analyst Ratings, and Sector Performance")
    print("✓ Ready for machine learning model training!")

    return split_data, metadata


if __name__ == "__main__":
    data, metadata = main()

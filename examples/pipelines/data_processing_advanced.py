"""
Advanced Data Processing Pipeline Example.

This example demonstrates advanced DataProcessor features:
- Multiple indicator configurations
- Custom indicator parameters
- Parameter combinations for grid search
- Train/test splitting
- Metadata tracking

No API keys required - uses Yahoo Finance.
"""

from datetime import datetime, timedelta

from quantrl_lab.data import DataProcessor, YFinanceDataLoader


def main():
    print("=" * 80)
    print("Advanced Data Processing Pipeline Example")
    print("=" * 80)

    # Step 1: Load OHLCV data
    print("\n[1/4] Loading OHLCV data...")
    loader = YFinanceDataLoader()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data

    ohlcv_df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timeframe="1d",
    )

    print(f"✓ Loaded {len(ohlcv_df)} rows")

    # Step 2: Configure advanced indicators
    print("\n[2/4] Configuring advanced indicator settings...")

    # Multiple configurations using list-of-values and parameter combinations
    indicators = [
        # Multiple windows defined as a list
        {"SMA": {"window": [10, 20, 50]}},
        {"EMA": {"window": [12, 26]}},
        {"RSI": {"window": [14, 21]}},
        # Custom MACD parameters
        {"MACD": {"fast": 12, "slow": 26, "signal": 9}},
        # This will generate all combinations of window AND num_std (2x2 = 4 sets)
        {"BB": {"window": [10, 20], "num_std": [1.5, 2.0]}},
        "ATR",
        "STOCH",
    ]

    print(f"✓ Configured {len(indicators)} indicator configurations")

    # Step 3: Process with train/test split
    print("\n[3/4] Processing data with train/test split...")

    processor = DataProcessor(ohlcv_data=ohlcv_df)

    # Split data: 70% train, 30% test
    split_config = {
        "train": 0.7,
        "test": 0.3,
    }

    split_data, metadata = processor.data_processing_pipeline(
        indicators=indicators,
        split_config=split_config,
    )

    print("✓ Data processed and split")
    print(f"  Train set: {split_data['train'].shape}")
    print(f"  Test set: {split_data['test'].shape}")
    print(
        f"  Train date range: {metadata['date_ranges']['train']['start']} to {metadata['date_ranges']['train']['end']}"
    )
    print(f"  Test date range: {metadata['date_ranges']['test']['start']} to {metadata['date_ranges']['test']['end']}")

    # Step 4: Analyze results
    print("\n[4/4] Analysis of processed data:")
    print("-" * 80)

    train_df = split_data["train"]
    test_df = split_data["test"]

    # Show feature count
    feature_cols = [col for col in train_df.columns if col not in ["Date", "Symbol", "Timestamp"]]
    print(f"Total features: {len(feature_cols)}")

    # Group features by type
    sma_features = [col for col in feature_cols if col.startswith("SMA_")]
    ema_features = [col for col in feature_cols if col.startswith("EMA_")]
    rsi_features = [col for col in feature_cols if col.startswith("RSI_")]
    bb_features = [col for col in feature_cols if col.startswith("BB_")]
    macd_features = [col for col in feature_cols if col.startswith("MACD")]
    other_features = [
        col for col in feature_cols if col.startswith(("Open", "High", "Low", "Close", "Volume", "ATR", "STOCH"))
    ]

    print("\nFeature breakdown:")
    print(f"  SMA features: {len(sma_features)} - {sma_features}")
    print(f"  EMA features: {len(ema_features)} - {ema_features}")
    print(f"  RSI features: {len(rsi_features)} - {rsi_features}")
    print(f"  Bollinger Bands: {len(bb_features)} - {bb_features}")
    print(f"  MACD features: {len(macd_features)} - {macd_features}")
    print(f"  Other features: {len(other_features)}")

    # Show correlation with Close price
    print("\n" + "=" * 80)
    print("Feature Correlation with Close Price (Train Set)")
    print("=" * 80)

    correlations = train_df.corr()["Close"].sort_values(ascending=False)
    print(correlations.head(15))

    # Data quality check
    print("\n" + "=" * 80)
    print("Data Quality Check")
    print("=" * 80)
    print(f"Train set missing values: {train_df.isna().sum().sum()}")
    print(f"Test set missing values: {test_df.isna().sum().sum()}")
    print(f"Train set infinite values: {train_df.isin([float('inf'), float('-inf')]).sum().sum()}")
    print(f"Test set infinite values: {test_df.isin([float('inf'), float('-inf')]).sum().sum()}")

    print("\n✓ Advanced processing complete!")
    print(f"  Ready for model training with {len(feature_cols)} features")

    return split_data, metadata


if __name__ == "__main__":
    data, metadata = main()

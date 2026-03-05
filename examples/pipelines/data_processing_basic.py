"""
Basic Data Processing Pipeline Example.

This example demonstrates the fundamental usage of DataProcessor to:
- Load OHLCV data
- Apply technical indicators
- Process and prepare data for ML models

No API keys required - uses Yahoo Finance.
"""

from datetime import datetime, timedelta

from quantrl_lab.data import DataProcessor, YFinanceDataLoader


def main():
    print("=" * 80)
    print("Basic Data Processing Pipeline Example")
    print("=" * 80)

    # Step 1: Load OHLCV data
    print("\n[1/3] Loading OHLCV data from Yahoo Finance...")
    loader = YFinanceDataLoader()

    # Get last 6 months of daily data for Apple
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    ohlcv_df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timeframe="1d",
    )

    print(f"✓ Loaded {len(ohlcv_df)} rows of OHLCV data")
    print(f"  Columns: {list(ohlcv_df.columns)}")
    print(f"  Date range: {ohlcv_df['Date'].min()} to {ohlcv_df['Date'].max()}")

    # Step 2: Create processor and apply indicators
    print("\n[2/3] Creating DataProcessor and applying technical indicators...")
    processor = DataProcessor(ohlcv_data=ohlcv_df)

    # Define indicators to apply
    indicators = [
        "SMA",  # Simple Moving Average (default window=20)
        "EMA",  # Exponential Moving Average (default window=20)
        "RSI",  # Relative Strength Index (default window=14)
        "MACD",  # Moving Average Convergence Divergence
        {"BB": {"window": 20, "num_std": 2}},  # Bollinger Bands with custom params
    ]

    # Process data
    processed_data, metadata = processor.data_processing_pipeline(
        indicators=indicators,
        # Don't drop Date column yet - we'll use it for display
        columns_to_drop=["Timestamp", "Symbol"],
    )

    print(f"✓ Applied {len(metadata['technical_indicators'])} indicator types")
    print(f"  Original shape: {metadata['original_shape']}")
    print(f"  Final shape: {metadata['final_shapes']['full_data']}")
    print(f"  Columns dropped: {metadata['columns_dropped']}")

    # Step 3: Display results
    print("\n[3/3] Processing complete! Sample of processed data:")
    print("-" * 80)

    # Show first few rows with key columns
    display_cols = ["Date", "Close", "SMA_20", "EMA_20", "RSI_14", "MACD", "BB_upper", "BB_lower"]
    available_cols = [col for col in display_cols if col in processed_data.columns]

    print(processed_data[available_cols].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(processed_data[["Close", "SMA_20", "RSI_14"]].describe())

    print("\n✓ Data is ready for machine learning!")
    print(f"  Shape: {processed_data.shape}")
    print(f"  Features: {len(processed_data.columns)} columns")
    print(f"  No missing values: {processed_data.isna().sum().sum() == 0}")

    return processed_data, metadata


if __name__ == "__main__":
    data, metadata = main()

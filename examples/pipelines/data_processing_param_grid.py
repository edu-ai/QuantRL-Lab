"""
Data Processing with Parameter Grid Search.

This example demonstrates:
- Using parameter combinations for indicator optimization
- Creating multiple feature sets automatically
- Preparing data for hyperparameter tuning
- Three-way train/val/test split

No API keys required - uses Yahoo Finance.
"""

from datetime import datetime, timedelta

from quantrl_lab.data import DataProcessor, YFinanceDataLoader


def main():
    print("=" * 80)
    print("Data Processing with Parameter Grid Search Example")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/4] Loading OHLCV data...")
    loader = YFinanceDataLoader()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years of data

    ohlcv_df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timeframe="1d",
    )

    print(f"✓ Loaded {len(ohlcv_df)} rows")

    # Step 2: Define parameter grid
    print("\n[2/4] Defining parameter grid for indicators...")

    # Use list-of-values syntax for automatic generation
    indicators_list = [
        # This one entry replaces 3 separate dicts
        {"SMA": {"window": [10, 20, 50]}},
        # This one entry replaces 2 separate dicts
        {"RSI": {"window": [14, 21]}},
    ]

    print(f"\n✓ Configured {len(indicators_list)} elegant indicator entries (generating 5 total feature sets)")

    # Step 3: Process with list-based parameters
    print("\n[3/4] Processing with list-based parameters...")

    processor = DataProcessor(ohlcv_data=ohlcv_df)

    # Three-way split: train/validation/test
    split_config = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    }

    split_data, metadata = processor.data_processing_pipeline(
        indicators=indicators_list,
        split_config=split_config,
    )

    print("✓ Data processed with 3-way split")
    print(f"  Train: {split_data['train'].shape}")
    print(f"  Val: {split_data['val'].shape}")
    print(f"  Test: {split_data['test'].shape}")

    # Step 4: Alternative - Parameter combination format
    print("\n[4/4] Demonstrating parameter combination format...")
    print("(This generates multiple features from parameter combinations)")

    # Single indicator with multiple parameter values
    # This will create SMA_10, SMA_20, and SMA_50 columns
    indicators_kwargs = ["SMA", "RSI"]

    # Pass parameters via kwargs
    result_combo, metadata_combo = processor.data_processing_pipeline(
        indicators=indicators_kwargs,
        SMA_params=[
            {"window": 10},
            {"window": 20},
            {"window": 50},
        ],
        RSI_params=[
            {"window": 14},
            {"window": 21},
        ],
    )

    print("✓ Alternative approach processed")
    print(f"  Shape: {result_combo.shape}")

    # Compare the two approaches
    print("\n" + "=" * 80)
    print("Feature Comparison")
    print("=" * 80)

    # Get SMA and RSI features
    sma_features_list = [col for col in split_data["train"].columns if col.startswith("SMA_")]
    rsi_features_list = [col for col in split_data["train"].columns if col.startswith("RSI_")]

    sma_features_combo = [col for col in result_combo.columns if col.startswith("SMA_")]
    rsi_features_combo = [col for col in result_combo.columns if col.startswith("RSI_")]

    print("List format features:")
    print(f"  SMA: {sma_features_list}")
    print(f"  RSI: {rsi_features_list}")
    print(f"  Total: {len(sma_features_list) + len(rsi_features_list)}")

    print("\nKwargs format features:")
    print(f"  SMA: {sma_features_combo}")
    print(f"  RSI: {rsi_features_combo}")
    print(f"  Total: {len(sma_features_combo) + len(rsi_features_combo)}")

    print("\n" + "=" * 80)
    print("Date Ranges for Each Split")
    print("=" * 80)
    for split_name, date_range in metadata["date_ranges"].items():
        print(f"{split_name.capitalize():>10}: {date_range['start']} to {date_range['end']}")

    # Demonstrate temporal ordering
    print("\n" + "=" * 80)
    print("Temporal Ordering Verification")
    print("=" * 80)

    from datetime import datetime as dt

    train_end = dt.strptime(metadata["date_ranges"]["train"]["end"], "%Y-%m-%d")
    val_start = dt.strptime(metadata["date_ranges"]["val"]["start"], "%Y-%m-%d")
    val_end = dt.strptime(metadata["date_ranges"]["val"]["end"], "%Y-%m-%d")
    test_start = dt.strptime(metadata["date_ranges"]["test"]["start"], "%Y-%m-%d")

    print(f"✓ Train ends before val starts: {train_end < val_start}")
    print(f"✓ Val ends before test starts: {val_end < test_start}")
    print("  This prevents data leakage in time-series analysis")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("List format: Best for explicit parameter sets")
    print("  - Clear and explicit")
    print("  - Easy to read and maintain")
    print("  - Recommended for most use cases")
    print("\nKwargs format: Best for quick parameter variations")
    print("  - More concise for testing parameter ranges")
    print("  - Good for prototyping")
    print("\nBoth approaches produce the same features!")

    print("\n✓ Ready for hyperparameter tuning with train/val/test splits!")

    return split_data, metadata


if __name__ == "__main__":
    data, metadata = main()

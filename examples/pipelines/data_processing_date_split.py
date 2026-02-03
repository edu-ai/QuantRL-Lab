"""
Data Processing with Date-Based Splitting.

This example demonstrates:
- Date-based train/test splitting (vs ratio-based)
- Custom date ranges for each split
- Walk-forward validation setup
- Metadata tracking for date ranges

No API keys required - uses Yahoo Finance.
"""

from datetime import datetime, timedelta

from quantrl_lab.data import DataProcessor, YFinanceDataLoader


def main():
    print("=" * 80)
    print("Data Processing with Date-Based Splitting Example")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/3] Loading OHLCV data...")
    loader = YFinanceDataLoader()

    # Load 3 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # ~3 years

    ohlcv_df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        timeframe="1d",
    )

    print(f"✓ Loaded {len(ohlcv_df)} rows")
    print(f"  Date range: {ohlcv_df['Date'].min()} to {ohlcv_df['Date'].max()}")

    # Step 2: Define date-based splits
    print("\n[2/3] Defining date-based splits...")

    # Calculate split dates
    total_days = (end_date - start_date).days
    train_end_date = start_date + timedelta(days=int(total_days * 0.6))
    val_end_date = start_date + timedelta(days=int(total_days * 0.8))

    print(f"  Training period: {start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
    print(f"  Validation period: {train_end_date.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}")
    print(f"  Test period: {val_end_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Create date-based split configuration
    split_config = {
        "train": (start_date.strftime("%Y-%m-%d"), train_end_date.strftime("%Y-%m-%d")),
        "val": (train_end_date.strftime("%Y-%m-%d"), val_end_date.strftime("%Y-%m-%d")),
        "test": (val_end_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    }

    # Step 3: Process data
    print("\n[3/3] Processing with date-based splits...")

    processor = DataProcessor(ohlcv_data=ohlcv_df)

    indicators = [
        {"SMA": {"window": 10}},
        {"SMA": {"window": 20}},
        {"SMA": {"window": 50}},
        "RSI",
        "MACD",
        "BB",
    ]

    split_data, metadata = processor.data_processing_pipeline(
        indicators=indicators,
        split_config=split_config,
    )

    print("✓ Data processed with date-based splits")

    # Display results
    print("\n" + "=" * 80)
    print("Split Results")
    print("=" * 80)

    for split_name in ["train", "val", "test"]:
        print(f"\n{split_name.upper()} Set:")
        print(f"  Shape: {split_data[split_name].shape}")
        print(
            f"  Date range: {metadata['date_ranges'][split_name]['start']} "
            f"to {metadata['date_ranges'][split_name]['end']}"
        )
        print(f"  Number of rows: {metadata['final_shapes'][split_name][0]}")
        print(f"  Number of features: {metadata['final_shapes'][split_name][1]}")

    # Calculate percentage of data in each split
    total_rows = sum(metadata['final_shapes'][split][0] for split in metadata['final_shapes'])
    print("\n" + "=" * 80)
    print("Data Distribution")
    print("=" * 80)
    for split_name in ["train", "val", "test"]:
        rows = metadata['final_shapes'][split_name][0]
        pct = (rows / total_rows) * 100
        print(f"{split_name.capitalize():>10}: {rows:4d} rows ({pct:5.1f}%)")

    # Show advantages of date-based splitting
    print("\n" + "=" * 80)
    print("Advantages of Date-Based Splitting")
    print("=" * 80)
    print("✓ Explicit control over date ranges")
    print("✓ Perfect for walk-forward validation")
    print("✓ Easy to align with market events (earnings, crises, etc.)")
    print("✓ Reproducible across different data sources")
    print("✓ No data leakage concerns")

    # Demonstrate walk-forward validation setup
    print("\n" + "=" * 80)
    print("Walk-Forward Validation Example")
    print("=" * 80)
    print("You can create multiple date-based splits for walk-forward testing:")

    # Example walk-forward windows
    window_size_days = 365  # 1 year training window
    test_size_days = 90  # 3 months test window

    walk_forward_example = []
    current_date = start_date

    window_num = 1
    while current_date + timedelta(days=window_size_days + test_size_days) <= end_date:
        train_start = current_date
        train_end = current_date + timedelta(days=window_size_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_size_days)

        walk_forward_example.append(
            {
                "window": window_num,
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
            }
        )

        current_date += timedelta(days=test_size_days)  # Slide window
        window_num += 1

    print(f"\nExample: {len(walk_forward_example)} walk-forward windows")
    for i, window in enumerate(walk_forward_example[:3], 1):  # Show first 3
        print(f"\n  Window {window['window']}:")
        print(f"    Train: {window['train'][0]} to {window['train'][1]}")
        print(f"    Test:  {window['test'][0]} to {window['test'][1]}")

    if len(walk_forward_example) > 3:
        print(f"  ... and {len(walk_forward_example) - 3} more windows")

    print("\n✓ Date-based splitting complete!")
    print("  Perfect for time-series validation strategies")

    return split_data, metadata


if __name__ == "__main__":
    data, metadata = main()

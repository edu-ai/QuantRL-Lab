"""
Example script demonstrating how to use the Indicator Registry.

This script shows:
1. How to fetch data (using yfinance for this example).
2. How to apply built-in technical indicators from the registry.
3. How to create and register a CUSTOM indicator on the fly.
4. How to use the custom indicator just like a built-in one.
"""

import pandas as pd
from loguru import logger

from quantrl_lab.data.indicators import IndicatorRegistry
from quantrl_lab.data.sources.yfinance_loader import YFinanceLoader


def main():
    # 1. Fetch some sample data
    logger.info("Fetching sample data (AAPL) from YFinance...")
    loader = YFinanceLoader()
    df = loader.load_data(symbols=["AAPL"], start_date="2023-01-01", end_date="2023-06-01")

    # Basic data check
    logger.info(f"Data loaded: {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")

    # ---------------------------------------------------------
    # 2. Apply Built-in Indicators
    # ---------------------------------------------------------
    logger.info("\n--- Applying Built-in Indicators ---")

    # You can list all available indicators
    available_indicators = IndicatorRegistry.list_all()
    logger.info(f"Available indicators: {available_indicators}")

    # Apply Simple Moving Average (SMA)
    logger.info("Applying SMA (window=20)...")
    df = IndicatorRegistry.apply("SMA", df, window=20)

    # Apply RSI
    logger.info("Applying RSI (window=14)...")
    df = IndicatorRegistry.apply("RSI", df, window=14)

    # Apply Bollinger Bands
    logger.info("Applying Bollinger Bands (window=20, num_std=2)...")
    df = IndicatorRegistry.apply("BB", df, window=20, num_std=2.0)

    # Check the new columns
    logger.info(f"Columns after built-ins: {df.columns.tolist()}")
    logger.info(f"Last row preview:\n{df.iloc[-1][['Close', 'SMA_20', 'RSI_14', 'BB_upper_20_2.0']]}")

    # ---------------------------------------------------------
    # 3. Register a Custom Indicator on the Fly
    # ---------------------------------------------------------
    logger.info("\n--- Registering Custom Indicator ---")

    # Let's define a custom indicator: "Price Rate of Change" (ROC)
    # ROC = ((Current Price / Price n periods ago) - 1) * 100

    # The decorator registers it automatically!
    @IndicatorRegistry.register(name="ROC")
    def rate_of_change(df: pd.DataFrame, window: int = 12, column: str = "Close") -> pd.DataFrame:
        """
        Calculate Rate of Change (ROC).

        Args:
            df (pd.DataFrame): Input dataframe
            window (int): Lookback period
            column (str): Column to calculate on

        Returns:
            pd.DataFrame: Dataframe with ROC column added
        """
        result = df.copy()

        # Define calculation logic
        def calc_roc(x):
            return ((x / x.shift(window)) - 1) * 100

        # Handle multi-symbol logic if your data structure supports it
        if "Symbol" in result.columns:
            result[f"ROC_{window}"] = result.groupby("Symbol")[column].transform(calc_roc)
        else:
            result[f"ROC_{window}"] = calc_roc(result[column])

        return result

    logger.info("Successfully registered 'ROC' indicator.")
    logger.info(f"Updated registry: {IndicatorRegistry.list_all()}")

    # ---------------------------------------------------------
    # 4. Use the Custom Indicator
    # ---------------------------------------------------------
    logger.info("Applying Custom ROC Indicator (window=12)...")

    # Now we can use .apply() with our new name "ROC"
    df = IndicatorRegistry.apply("ROC", df, window=12)

    logger.info(f"Columns after custom indicator: {df.columns.tolist()}")

    # verify values exist
    last_val = df.iloc[-1]["ROC_12"]
    logger.info(f"Calculated ROC_12 for last row: {last_val:.4f}%")


if __name__ == "__main__":
    main()

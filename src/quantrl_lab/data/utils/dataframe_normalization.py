"""
DataFrame normalization utilities for standardizing OHLCV data across
data sources.

This module provides unified functions to eliminate column renaming and
data transformation duplication across different data source
implementations.
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


def standardize_ohlcv_columns(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    drop_unmapped: bool = False,
) -> pd.DataFrame:
    """
    Rename columns in a DataFrame to standard OHLCV format.

    Args:
        df: Input DataFrame with provider-specific column names
        column_mapping: Dictionary mapping source columns to standard names
        drop_unmapped: If True, drop columns not in the mapping

    Returns:
        pd.DataFrame: DataFrame with standardized column names

    Examples:
        >>> mapping = {'date': 'Timestamp', 'open': 'Open', 'close': 'Close'}
        >>> df = standardize_ohlcv_columns(df, mapping)
    """
    df_copy = df.copy()
    df_copy.rename(columns=column_mapping, inplace=True)

    if drop_unmapped:
        # Keep only columns that are values in the mapping (standardized names)
        standard_columns = set(column_mapping.values())
        existing_standard_columns = [col for col in df_copy.columns if col in standard_columns]
        df_copy = df_copy[existing_standard_columns]

    return df_copy


def add_symbol_column(
    df: pd.DataFrame,
    symbol: str,
    position: str = "end",
) -> pd.DataFrame:
    """
    Add a Symbol column to a DataFrame.

    Args:
        df: Input DataFrame
        symbol: Symbol string to add
        position: Where to place the column ('start' or 'end')

    Returns:
        pd.DataFrame: DataFrame with Symbol column added

    Examples:
        >>> df = add_symbol_column(df, 'AAPL', position='start')
    """
    df_copy = df.copy()
    df_copy["Symbol"] = symbol

    if position == "start":
        # Move Symbol to first column
        cols = ["Symbol"] + [col for col in df_copy.columns if col != "Symbol"]
        df_copy = df_copy[cols]

    return df_copy


def add_date_column_from_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Add a Date column extracted from a Timestamp column.

    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of the timestamp column
        date_col: Name for the new date-only column

    Returns:
        pd.DataFrame: DataFrame with Date column added

    Raises:
        ValueError: If timestamp column doesn't exist

    Examples:
        >>> df = add_date_column_from_timestamp(df)
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

    df_copy = df.copy()

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_col]):
        df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

    # Extract date only
    df_copy[date_col] = df_copy[timestamp_col].dt.date

    return df_copy


def sort_by_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    ascending: bool = True,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Sort DataFrame by timestamp column.

    Args:
        df: Input DataFrame
        timestamp_col: Name of the timestamp column to sort by
        ascending: Sort order (True = oldest first, False = newest first)
        reset_index: If True, reset the index after sorting

    Returns:
        pd.DataFrame: Sorted DataFrame

    Raises:
        ValueError: If timestamp column doesn't exist

    Examples:
        >>> df = sort_by_timestamp(df, ascending=True)
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

    df_copy = df.copy()
    df_copy.sort_values(timestamp_col, ascending=ascending, inplace=True)

    if reset_index:
        df_copy.reset_index(drop=True, inplace=True)

    return df_copy


def convert_columns_to_numeric(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Convert specified columns to numeric types.

    Args:
        df: Input DataFrame
        columns: List of column names to convert (None = auto-detect OHLCV columns)
        errors: How to handle conversion errors ('coerce', 'raise', 'ignore')

    Returns:
        pd.DataFrame: DataFrame with numeric columns converted

    Examples:
        >>> df = convert_columns_to_numeric(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
    """
    df_copy = df.copy()

    # Auto-detect standard OHLCV columns if not specified
    if columns is None:
        standard_ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj_close"]
        columns = [col for col in standard_ohlcv_cols if col in df_copy.columns]

    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors=errors)

    return df_copy


def standardize_ohlcv_dataframe(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    symbol: Optional[str] = None,
    timestamp_col: str = "Timestamp",
    add_date: bool = True,
    sort_data: bool = True,
    convert_numeric: bool = True,
) -> pd.DataFrame:
    """
    Apply full standardization pipeline to OHLCV data.

    This is a convenience function that combines multiple standardization steps:
    1. Rename columns to standard format
    2. Add Symbol column (if symbol provided)
    3. Add Date column from Timestamp
    4. Sort by Timestamp
    5. Convert OHLCV columns to numeric

    Args:
        df: Input DataFrame
        column_mapping: Dictionary mapping source columns to standard names
        symbol: Optional symbol to add as a column
        timestamp_col: Name of timestamp column after mapping
        add_date: If True, add Date column from Timestamp
        sort_data: If True, sort by timestamp
        convert_numeric: If True, convert OHLCV columns to numeric

    Returns:
        pd.DataFrame: Fully standardized DataFrame

    Examples:
        >>> mapping = {'date': 'Timestamp', 'open': 'Open', 'close': 'Close'}
        >>> df = standardize_ohlcv_dataframe(df, mapping, symbol='AAPL')
    """
    # Step 1: Rename columns
    df_result = standardize_ohlcv_columns(df, column_mapping)

    # Step 2: Ensure timestamp is datetime
    if timestamp_col in df_result.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_result[timestamp_col]):
            df_result[timestamp_col] = pd.to_datetime(df_result[timestamp_col])

    # Step 3: Add Symbol column
    if symbol is not None:
        df_result = add_symbol_column(df_result, symbol)

    # Step 4: Add Date column
    if add_date and timestamp_col in df_result.columns:
        df_result = add_date_column_from_timestamp(df_result, timestamp_col)

    # Step 5: Sort by timestamp
    if sort_data and timestamp_col in df_result.columns:
        df_result = sort_by_timestamp(df_result, timestamp_col)

    # Step 6: Convert to numeric
    if convert_numeric:
        df_result = convert_columns_to_numeric(df_result)

    logger.debug(
        "Standardized DataFrame: {rows} rows, {cols} columns",
        rows=len(df_result),
        cols=len(df_result.columns),
    )

    return df_result

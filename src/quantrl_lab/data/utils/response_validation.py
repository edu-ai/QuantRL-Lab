"""
Response validation utilities for API data handling.

This module provides unified functions for validating API responses and
handling empty or invalid data consistently across all data sources.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger


def validate_api_response(
    response: Any,
    expected_type: type,
    min_length: Optional[int] = None,
    allow_empty: bool = False,
    error_message: Optional[str] = None,
) -> bool:
    """
    Validate an API response meets basic expectations.

    Args:
        response: The API response to validate
        expected_type: Expected type (list, dict, etc.)
        min_length: Minimum length for list/dict responses
        allow_empty: If True, allow empty responses
        error_message: Custom error message prefix

    Returns:
        bool: True if valid, False otherwise

    Examples:
        >>> is_valid = validate_api_response(data, list, min_length=1)
    """
    prefix = error_message or "API response validation failed"

    # Check None
    if response is None:
        if not allow_empty:
            logger.warning(f"{prefix}: Response is None")
            return False
        return True

    # Check type
    if not isinstance(response, expected_type):
        logger.warning(f"{prefix}: Expected {expected_type.__name__}, got {type(response).__name__}")
        return False

    # Check length for collections
    if min_length is not None:
        if hasattr(response, '__len__'):
            if len(response) < min_length:
                logger.warning(f"{prefix}: Expected minimum length {min_length}, got {len(response)}")
                return False

    # Check empty
    if not allow_empty:
        if hasattr(response, '__len__') and len(response) == 0:
            logger.warning(f"{prefix}: Response is empty")
            return False

    return True


def convert_to_dataframe_safe(
    data: Union[List[Dict], Dict],
    expected_min_rows: int = 0,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Safely convert API response data to DataFrame with validation.

    Args:
        data: API response data (list of dicts or dict)
        expected_min_rows: Minimum expected rows (0 = no minimum)
        symbol: Optional symbol for logging context

    Returns:
        pd.DataFrame: Converted DataFrame (empty if validation fails)

    Examples:
        >>> df = convert_to_dataframe_safe(api_data, expected_min_rows=1, symbol='AAPL')
    """
    context = f"for symbol: {symbol}" if symbol else ""

    # Validate input
    if not validate_api_response(data, (list, dict), allow_empty=(expected_min_rows == 0)):
        logger.warning(f"Invalid API response {context}")
        return pd.DataFrame()

    # Handle empty responses
    if isinstance(data, list) and len(data) == 0:
        logger.warning(f"Empty response list {context}")
        return pd.DataFrame()

    if isinstance(data, dict) and len(data) == 0:
        logger.warning(f"Empty response dict {context}")
        return pd.DataFrame()

    # Convert to DataFrame
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to convert response to DataFrame {context}: {e}")
        return pd.DataFrame()

    # Check if DataFrame is empty
    if df.empty:
        logger.warning(f"Conversion resulted in empty DataFrame {context}")
        return pd.DataFrame()

    # Check minimum rows
    if expected_min_rows > 0 and len(df) < expected_min_rows:
        logger.warning(f"DataFrame has {len(df)} rows, expected at least {expected_min_rows} {context}")

    logger.debug(f"Successfully converted response to DataFrame with {len(df)} rows {context}")
    return df


def check_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_on_missing: bool = False,
) -> bool:
    """
    Check if DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        raise_on_missing: If True, raise ValueError on missing columns

    Returns:
        bool: True if all required columns present

    Raises:
        ValueError: If raise_on_missing=True and columns are missing

    Examples:
        >>> has_cols = check_required_columns(df, ['Open', 'High', 'Low', 'Close'])
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        msg = f"Missing required columns: {missing_columns}. Available: {list(df.columns)}"

        if raise_on_missing:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    return True


def log_dataframe_info(
    df: pd.DataFrame,
    operation: str,
    symbol: Optional[str] = None,
    level: str = "success",
) -> None:
    """
    Log standardized information about a DataFrame result.

    Args:
        df: DataFrame to log info about
        operation: Description of the operation (e.g., "Fetched OHLCV data")
        symbol: Optional symbol for context
        level: Logging level ('debug', 'info', 'success', 'warning')

    Examples:
        >>> log_dataframe_info(df, "Fetched historical data", symbol='AAPL')
    """
    context = f" for {symbol}" if symbol else ""
    message = f"{operation}{context}: {len(df)} rows"

    if df.empty:
        logger.warning(f"{operation}{context}: returned empty DataFrame")
        return

    # Add column info for debug
    if level == "debug":
        message += f", {len(df.columns)} columns"

    # Log at appropriate level
    log_func = getattr(logger, level, logger.info)
    log_func(message)


def validate_date_range_data(
    df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    timestamp_col: str = 'Timestamp',
) -> bool:
    """
    Validate that DataFrame data falls within expected date range.

    Args:
        df: DataFrame with timestamp column
        start_date: Expected start date
        end_date: Expected end date
        timestamp_col: Name of timestamp column

    Returns:
        bool: True if data is within range (or no range specified)

    Examples:
        >>> is_valid = validate_date_range_data(df, start_date='2023-01-01', end_date='2023-12-31')
    """
    if df.empty:
        return True  # Empty data is technically in range

    if timestamp_col not in df.columns:
        logger.warning(f"Timestamp column '{timestamp_col}' not found for date range validation")
        return False

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            logger.warning(f"Failed to convert timestamp column to datetime: {e}")
            return False

    actual_start = df[timestamp_col].min()
    actual_end = df[timestamp_col].max()

    # Check start date
    if start_date is not None:
        expected_start = pd.to_datetime(start_date)
        if actual_start < expected_start:
            logger.warning(f"Data starts at {actual_start}, earlier than expected {expected_start}")

    # Check end date
    if end_date is not None:
        expected_end = pd.to_datetime(end_date)
        if actual_end > expected_end:
            logger.warning(f"Data ends at {actual_end}, later than expected {expected_end}")

    logger.debug(f"Data range validated: {actual_start} to {actual_end}")
    return True

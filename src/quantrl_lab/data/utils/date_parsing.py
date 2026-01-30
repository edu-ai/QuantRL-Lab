"""
Date parsing and normalization utilities for data sources.

This module provides unified date handling to eliminate duplication
across different data source implementations.
"""

from datetime import datetime
from typing import Optional, Tuple, Union

import dateutil.parser
import pandas as pd


def normalize_date(
    date: Union[str, datetime, pd.Timestamp],
    default_if_none: Optional[datetime] = None,
) -> datetime:
    """
    Normalize a date input to a datetime object.

    Args:
        date: Date as string, datetime, or pandas Timestamp
        default_if_none: Default datetime to return if date is None

    Returns:
        datetime: Normalized datetime object

    Raises:
        ValueError: If date cannot be parsed

    Examples:
        >>> normalize_date("2023-01-01")
        datetime.datetime(2023, 1, 1, 0, 0)
        >>> normalize_date(None, default_if_none=datetime.now())
        datetime.datetime(...)
    """
    if date is None:
        if default_if_none is not None:
            return default_if_none
        raise ValueError("Date cannot be None without a default value")

    if isinstance(date, datetime):
        return date

    if isinstance(date, pd.Timestamp):
        return date.to_pydatetime()

    if isinstance(date, str):
        try:
            # Use dateutil for flexible parsing
            return dateutil.parser.parse(date)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date string '{date}': {e}")

    raise TypeError(f"Unsupported date type: {type(date)}")


def normalize_date_range(
    start: Union[str, datetime, None],
    end: Union[str, datetime, None] = None,
    default_end_to_now: bool = True,
    validate_order: bool = True,
) -> Tuple[datetime, datetime]:
    """
    Normalize and validate a date range.

    Args:
        start: Start date as string or datetime
        end: End date as string or datetime (defaults to now if None)
        default_end_to_now: If True and end is None, default to datetime.now()
        validate_order: If True, validate that start <= end

    Returns:
        Tuple[datetime, datetime]: Normalized (start, end) datetime objects

    Raises:
        ValueError: If dates are invalid or start > end

    Examples:
        >>> start, end = normalize_date_range("2023-01-01", "2023-12-31")
        >>> start, end = normalize_date_range("2023-01-01")  # end defaults to now
    """
    if start is None:
        raise ValueError("Start date cannot be None")

    # Normalize start date
    start_dt = normalize_date(start)

    # Normalize end date
    if end is None:
        if default_end_to_now:
            end_dt = datetime.now()
        else:
            raise ValueError("End date cannot be None when default_end_to_now=False")
    else:
        end_dt = normalize_date(end)

    # Validate order
    if validate_order and start_dt > end_dt:
        raise ValueError(f"Start date ({start_dt}) must be before or equal to end date ({end_dt})")

    return start_dt, end_dt


def format_date_to_string(
    date: Union[str, datetime, pd.Timestamp],
    format_string: str = "%Y-%m-%d",
) -> str:
    """
    Format a date to a standardized string.

    Args:
        date: Date as string, datetime, or pandas Timestamp
        format_string: Output format string (default: YYYY-MM-DD)

    Returns:
        str: Formatted date string

    Raises:
        ValueError: If date cannot be parsed

    Examples:
        >>> format_date_to_string(datetime(2023, 1, 15))
        '2023-01-15'
        >>> format_date_to_string("2023-01-15T10:30:00", "%Y-%m-%d")
        '2023-01-15'
    """
    dt = normalize_date(date)
    return dt.strftime(format_string)

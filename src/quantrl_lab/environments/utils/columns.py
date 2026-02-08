from typing import List, Optional

import pandas as pd


def detect_column_index(df: pd.DataFrame, candidates: List[str]) -> Optional[int]:
    """
    Detect a column index from a list of candidates (case-insensitive).

    Args:
        df: The DataFrame to search.
        candidates: List of column names to look for.

    Returns:
        The index of the first matching column, or None if not found.
    """
    columns = df.columns.tolist()

    # Exact match first
    for candidate in candidates:
        if candidate in columns:
            return columns.index(candidate)

    # Case insensitive match
    columns_lower = [c.lower() for c in columns]
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower.index(candidate.lower())

    return None


def auto_detect_price_column(df: pd.DataFrame) -> int:
    """
    Auto-detect the price column index from a DataFrame using standard
    naming conventions.

    Args:
        df: Input DataFrame with price data.

    Returns:
        int: Index of the detected price column.

    Raises:
        ValueError: If no suitable price column is found.
    """
    columns = df.columns.tolist()

    # Priority order for price column detection
    price_candidates = [
        "close",
        "Close",
        "CLOSE",
        "price",
        "Price",
        "PRICE",
        "adj_close",
        "Adj Close",
        "ADJ_CLOSE",
        "adjusted_close",
        "Adjusted_Close",
    ]

    # Use the helper to find it
    idx = detect_column_index(df, price_candidates)

    if idx is not None:
        return idx

    # If no obvious price column found, check partial matches as fallback
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if any(candidate.lower() in col_lower for candidate in ["close", "price"]):
            return i

    raise ValueError(
        f"Could not auto-detect price column. Available columns: {columns}. "
        f"Please ensure your DataFrame has a column named 'close', 'price', or similar."
    )

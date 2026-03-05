"""
Symbol normalization and validation utilities for data sources.

This module provides unified symbol handling to eliminate duplication
across different data source implementations.
"""

from typing import List, Optional, Union

from loguru import logger


def normalize_symbols(
    symbols: Union[str, List[str]],
    max_symbols: Optional[int] = None,
    warn_on_limit: bool = True,
) -> List[str]:
    """
    Normalize symbol input to a list of strings.

    Args:
        symbols: Single symbol string or list of symbols
        max_symbols: Maximum number of symbols allowed (None = unlimited)
        warn_on_limit: If True, log warning when symbols exceed max_symbols

    Returns:
        List[str]: Normalized list of symbol strings

    Raises:
        TypeError: If symbols is not a string or list
        ValueError: If list contains non-string elements

    Examples:
        >>> normalize_symbols("AAPL")
        ['AAPL']
        >>> normalize_symbols(["AAPL", "GOOGL"])
        ['AAPL', 'GOOGL']
        >>> normalize_symbols(["AAPL", "GOOGL", "MSFT"], max_symbols=2)
        ['AAPL', 'GOOGL']  # with warning logged
    """
    # Convert single string to list
    if isinstance(symbols, str):
        symbol_list = [symbols]
    elif isinstance(symbols, list):
        # Validate all elements are strings
        if not all(isinstance(symbol, str) for symbol in symbols):
            raise ValueError("All elements in symbols list must be strings")
        symbol_list = symbols
    else:
        raise TypeError(f"symbols must be a string or list of strings, got {type(symbols)}")

    # Check max symbols limit
    if max_symbols is not None and len(symbol_list) > max_symbols:
        if warn_on_limit:
            logger.warning(
                "Received {count} symbols but only {max} are supported. Using first {max} symbols.",
                count=len(symbol_list),
                max=max_symbols,
            )
        symbol_list = symbol_list[:max_symbols]

    return symbol_list


def validate_symbols(
    symbols: Union[str, List[str]],
    max_symbols: Optional[int] = None,
    allow_empty: bool = False,
) -> None:
    """
    Validate symbol input without normalization.

    Args:
        symbols: Single symbol string or list of symbols
        max_symbols: Maximum number of symbols allowed (None = unlimited)
        allow_empty: If True, allow empty strings in symbol list

    Raises:
        TypeError: If symbols is not a string or list
        ValueError: If validation fails (empty list, non-string elements, etc.)

    Examples:
        >>> validate_symbols("AAPL")  # passes
        >>> validate_symbols(["AAPL", ""])  # raises ValueError
        >>> validate_symbols(["AAPL", ""], allow_empty=True)  # passes
        >>> validate_symbols(["AAPL", "GOOGL", "MSFT"], max_symbols=2)  # raises ValueError
    """
    # Type check
    if not isinstance(symbols, (str, list)):
        raise TypeError(f"symbols must be a string or list of strings, got {type(symbols)}")

    # Convert to list for validation
    if isinstance(symbols, str):
        symbol_list = [symbols]
    else:
        symbol_list = symbols

    # Check for empty list
    if not symbol_list:
        raise ValueError("symbols list cannot be empty")

    # Validate all elements are strings
    for i, symbol in enumerate(symbol_list):
        if not isinstance(symbol, str):
            raise ValueError(f"Element at index {i} is not a string: {type(symbol)}")

        # Check for empty strings
        if not allow_empty and not symbol.strip():
            raise ValueError(f"Symbol at index {i} is empty or whitespace-only")

    # Check max symbols
    if max_symbols is not None and len(symbol_list) > max_symbols:
        raise ValueError(f"Too many symbols: received {len(symbol_list)}, maximum allowed is {max_symbols}")


def get_single_symbol(
    symbols: Union[str, List[str]],
    warn_on_multiple: bool = True,
) -> str:
    """
    Extract a single symbol from string or list input.

    If a list is provided, returns the first symbol and optionally warns
    if the list contains multiple symbols.

    Args:
        symbols: Single symbol string or list of symbols
        warn_on_multiple: If True, log warning when multiple symbols provided

    Returns:
        str: Single symbol string

    Raises:
        TypeError: If symbols is not a string or list
        ValueError: If symbols list is empty

    Examples:
        >>> get_single_symbol("AAPL")
        'AAPL'
        >>> get_single_symbol(["AAPL", "GOOGL"])  # logs warning
        'AAPL'
        >>> get_single_symbol(["AAPL", "GOOGL"], warn_on_multiple=False)
        'AAPL'
    """
    symbol_list = normalize_symbols(symbols)

    if len(symbol_list) > 1 and warn_on_multiple:
        logger.warning(
            "Multiple symbols provided ({count}), using only the first symbol: {symbol}",
            count=len(symbol_list),
            symbol=symbol_list[0],
        )

    return symbol_list[0]

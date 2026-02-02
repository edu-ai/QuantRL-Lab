"""
Centralized configuration for the data module.

This module provides a single source of truth for all configuration
constants, defaults, and settings used throughout the data module.
"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class DataConfig:
    """
    Central configuration for data module.

    This dataclass consolidates all configuration constants and defaults
    used across the data module, making them easy to discover, modify,
    and override for testing.

    Attributes:
        DATE_COLUMNS: List of recognized date column names (case-sensitive)
        DEFAULT_DATE_COLUMN: Default name for date column when creating DataFrames
        OHLCV_COLUMNS_LOWER: Set of required OHLCV columns in lowercase
        OHLCV_COLUMNS_UPPER: Set of required OHLCV columns in uppercase (title case)
        DEFAULT_TIMEFRAME: Default timeframe for historical data requests
        MAX_RETRIES: Maximum number of retry attempts for API calls
        RETRY_BACKOFF_FACTOR: Exponential backoff multiplier for retries
        REQUEST_TIMEOUT: Default timeout for HTTP requests (seconds)
        DEFAULT_FILLNA_STRATEGY: Default strategy for filling missing values
        DEFAULT_SENTIMENT_MODEL: Default model for sentiment analysis
        CACHE_ENABLED: Whether to enable caching for data requests
        CACHE_TTL: Cache time-to-live in seconds
    """

    # Column name standards
    DATE_COLUMNS: List[str] = field(default_factory=lambda: ["Date", "date", "timestamp", "Timestamp"])
    DEFAULT_DATE_COLUMN: str = "Date"

    OHLCV_COLUMNS_LOWER: Set[str] = field(default_factory=lambda: {"open", "high", "low", "close", "volume"})
    OHLCV_COLUMNS_UPPER: Set[str] = field(default_factory=lambda: {"Open", "High", "Low", "Close", "Volume"})

    # API defaults
    DEFAULT_TIMEFRAME: str = "1d"
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 2.0
    REQUEST_TIMEOUT: int = 30  # seconds

    # Processing defaults
    DEFAULT_FILLNA_STRATEGY: str = "neutral"
    DEFAULT_SENTIMENT_MODEL: str = "finbert"

    # Caching settings
    CACHE_ENABLED: bool = False
    CACHE_TTL: int = 3600  # 1 hour in seconds

    def get_all_date_columns(self) -> List[str]:
        """
        Get all recognized date column names.

        Returns:
            List of all date column names (case variations)
        """
        return self.DATE_COLUMNS

    def is_date_column(self, column_name: str) -> bool:
        """
        Check if a column name is recognized as a date column.

        Args:
            column_name: Column name to check

        Returns:
            True if column_name is in DATE_COLUMNS
        """
        return column_name in self.DATE_COLUMNS

    def get_required_ohlcv_columns(self, case_sensitive: bool = True) -> Set[str]:
        """
        Get required OHLCV columns in specified case.

        Args:
            case_sensitive: If True, returns uppercase columns; if False, returns lowercase

        Returns:
            Set of required OHLCV column names
        """
        return self.OHLCV_COLUMNS_UPPER if case_sensitive else self.OHLCV_COLUMNS_LOWER

    def validate_ohlcv_columns(self, columns: List[str]) -> bool:
        """
        Check if a list of columns contains all required OHLCV columns.

        Args:
            columns: List of column names to validate

        Returns:
            True if all required OHLCV columns are present (case-insensitive)
        """
        columns_lower = {col.lower() for col in columns}
        return self.OHLCV_COLUMNS_LOWER.issubset(columns_lower)


# Global instance for easy importing
config = DataConfig()

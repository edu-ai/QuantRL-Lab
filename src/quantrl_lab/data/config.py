"""
Centralized configuration for the data module.

This module provides a single source of truth for all configuration
constants, defaults, and settings used throughout the data module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set

from pydantic import BaseModel, Field


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

# API Domains and Base URLs
FMP_DOMAIN = "https://financialmodelingprep.com"
FMP_API_BASE = f"{FMP_DOMAIN}/stable"

ALPHA_VANTAGE_DOMAIN = "https://www.alphavantage.co"
ALPHA_VANTAGE_API_BASE = f"{ALPHA_VANTAGE_DOMAIN}/query"


class FundamentalMetric(Enum):
    """Enum for available fundamental data metrics from Alpha
    Vantage."""

    COMPANY_OVERVIEW = ("company_overview", "Company overview and key statistics")
    # ETF_PROFILE = ("etf_profile", "ETF profile and holding information")
    DIVIDENDS = ("dividends", "Dividend payment history")
    SPLITS = ("splits", "Stock split history")
    INCOME_STATEMENT = ("income_statement", "Income statement data")
    BALANCE_SHEET = ("balance_sheet", "Balance sheet data")
    CASH_FLOW = ("cash_flow", "Cash flow statement data")
    EARNINGS = ("earnings", "Earnings data")

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj


class MacroIndicator(Enum):
    """Enumeration of macro economic indicators available from Alpha
    Vantage."""

    REAL_GDP = ("real_gdp", "Real Gross Domestic Product data")
    REAL_GDP_PER_CAPITA = ("real_gdp_per_capita", "Real GDP per capita data")
    TREASURY_YIELD = ("treasury_yield", "Treasury yield rates for various maturities")
    FEDERAL_FUNDS_RATE = ("federal_funds_rate", "Federal funds interest rate")
    CPI = ("cpi", "Consumer Price Index inflation data")
    INFLATION = ("inflation", "Annual inflation rate")
    RETAIL_SALES = ("retail_sales", "Monthly retail sales data")
    DURABLE_GOODS = ("durable_goods", "Durable goods orders data")
    UNEMPLOYMENT_RATE = ("unemployment_rate", "Monthly unemployment rate")
    NON_FARM_PAYROLL = ("non_farm_payroll", "Non-farm payroll employment data")

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj


class YFinanceInterval(str, Enum):
    """Valid intervals for YFinance data fetching."""

    MIN_1 = "1m"
    MIN_2 = "2m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    MIN_60 = "60m"
    MIN_90 = "90m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    DAY_5 = "5d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"
    MONTH_3 = "3mo"

    @classmethod
    def values(cls) -> list:
        return [interval.value for interval in cls]


class FinancialStatementType(str, Enum):
    """Types of financial statements."""

    INCOME = "income_statement"
    BALANCE = "balance_sheet"
    CASHFLOW = "cash_flow"


class FinancialColumnsConfig(BaseModel):
    """Essential financial data columns by statement type."""

    # Income statement columns
    income_columns: List[str] = Field(
        default=[
            "TotalRevenue",
            "GrossProfit",
            "OperatingIncome",
            "NetIncome",
            "EBITDA",
            "BasicEPS",
            "DilutedEPS",
            "OperatingExpense",
        ],
        description="Essential income statement columns",
    )

    # Balance sheet columns
    balance_columns: List[str] = Field(
        default=[
            "TotalAssets",
            "CurrentAssets",
            "TotalLiabilities",
            "CurrentLiabilities",
            "CashAndCashEquivalents",
            "TotalDebt",
            "TotalStockholderEquity",
        ],
        description="Essential balance sheet columns",
    )

    # Cash flow statement columns
    cashflow_columns: List[str] = Field(
        default=[
            "OperatingCashFlow",
            "InvestingCashFlow",
            "FinancingCashFlow",
            "CapitalExpenditures",
            "DividendPaid",
            "FreeCashFlow",
        ],
        description="Essential cash flow statement columns",
    )

    # Macro/Economic indicators of importance
    macro_indicators: List[str] = Field(
        default=["GDP", "CPI", "federalFunds", "retailSales", "totalNonfarmPayroll"],
        description="Important macro indicators",
    )

    def get_columns_by_statement_type(self, statement_type: FinancialStatementType) -> List[str]:
        """Get columns for a specific statement type."""
        if statement_type == FinancialStatementType.INCOME:
            return self.income_columns
        elif statement_type == FinancialStatementType.BALANCE:
            return self.balance_columns
        elif statement_type == FinancialStatementType.CASHFLOW:
            return self.cashflow_columns
        else:
            raise ValueError(f"Unknown statement type: {statement_type}")

    def get_all_statement_columns(self) -> List[str]:
        """Get all financial statement columns combined."""
        return list(set(self.income_columns + self.balance_columns + self.cashflow_columns))

    def get_macro_indicators(self) -> List[str]:
        """Get the list of important macro indicators."""
        return self.macro_indicators


financial_columns = FinancialColumnsConfig()

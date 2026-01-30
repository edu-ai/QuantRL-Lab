# Data Sources Guide

This guide provides comprehensive documentation for all data sources available in QuantRL-Lab, including capabilities, usage patterns, API limitations, and best practices.

## Table of Contents
- [Overview](#overview)
- [Quick Comparison](#quick-comparison)
- [Protocol-Based Architecture](#protocol-based-architecture)
- [Data Source Details](#data-source-details)
  - [Alpaca](#alpaca)
  - [Yahoo Finance (YFinance)](#yahoo-finance-yfinance)
  - [Alpha Vantage](#alpha-vantage)
  - [Financial Modeling Prep (FMP)](#financial-modeling-prep-fmp)
- [Data Utilities](#data-utilities)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

QuantRL-Lab supports multiple financial data providers through a unified, protocol-based interface. Each data source implements specific capability protocols, allowing for flexible, runtime-checkable feature detection.

**Key Design Principles:**
- **Protocol-based interfaces**: Data sources implement capability protocols (e.g., `HistoricalDataCapable`, `LiveDataCapable`)
- **Standardized output**: All sources return normalized pandas DataFrames with consistent column names
- **Automatic retry logic**: Built-in HTTP request wrapper with configurable retry strategies
- **Centralized utilities**: Shared modules for date parsing, symbol validation, and DataFrame normalization

**Recent Refactoring (Commit 09d7695):**
All data loaders now use centralized utility functions from `src/quantrl_lab/data/utils/` instead of duplicate inline logic, improving maintainability and consistency.

---

## Quick Comparison

### Capability Matrix

| Capability | Alpaca | YFinance | Alpha Vantage | FMP |
|------------|--------|----------|---------------|-----|
| **Historical OHLCV (Daily)** | ✅ | ✅ | ✅ (100 days free) | ✅ |
| **Intraday Data** | ✅ Multiple timeframes | ✅ Limited (30 days) | 🔒 Premium only | ✅ 5min-4hour |
| **Real-time Quotes/Trades** | ✅ | ❌ | ❌ | ❌ |
| **Streaming (WebSocket)** | ✅ | ❌ | ❌ | ❌ |
| **Fundamental Data** | ❌ | ✅ | ✅ | ❌ |
| **News Data** | ✅ | ❌ | ✅ Sentiment | ❌ |
| **Macroeconomic Indicators** | ❌ | ❌ | ✅ | ❌ |
| **Analyst Data** | ❌ | ❌ | ❌ | ✅ Grades/Ratings |
| **Multi-symbol Support** | ✅ | ✅ | ✅ | ⚠️ Single symbol |
| **API Key Required** | ✅ | ❌ | ✅ | ✅ |
| **Free Tier Available** | ✅ | ✅ | ✅ Limited | ✅ |

**Legend:**
- ✅ Fully supported
- ⚠️ Limited support
- 🔒 Premium/paid feature only
- ❌ Not available

### Protocol Implementation

Each data source implements different capability protocols:

```python
# AlpacaDataLoader
Protocols: HistoricalDataCapable, LiveDataCapable, StreamingCapable,
           NewsDataCapable, ConnectionManaged

# YFinanceDataLoader
Protocols: HistoricalDataCapable, FundamentalDataCapable

# AlphaVantageDataLoader
Protocols: HistoricalDataCapable, FundamentalDataCapable,
           MacroDataCapable, NewsDataCapable

# FMPDataSource
Protocols: HistoricalDataCapable, AnalystDataCapable
```

### Use Case Recommendations

| Use Case | Recommended Source | Reason |
|----------|-------------------|--------|
| **Quick prototyping** | YFinance | Free, no API key, good daily data |
| **Production trading** | Alpaca | Real-time data, streaming, reliable |
| **Fundamental analysis** | YFinance or Alpha Vantage | Comprehensive financial statements |
| **Macroeconomic research** | Alpha Vantage | Exclusive macro data (GDP, CPI, etc.) |
| **Intraday backtesting** | FMP or Alpaca | Multiple intraday timeframes |
| **Analyst sentiment** | FMP | Historical grades and ratings |
| **News sentiment** | Alpaca or Alpha Vantage | News with metadata/sentiment |

---

## Protocol-Based Architecture

### Capability Protocols

QuantRL-Lab uses Python protocols (structural typing) to define data source capabilities:

```python
from quantrl_lab.data.interface import (
    HistoricalDataCapable,
    LiveDataCapable,
    NewsDataCapable,
    FundamentalDataCapable,
    MacroDataCapable,
    AnalystDataCapable,
    StreamingCapable,
    ConnectionManaged,
)

# Runtime feature detection
from quantrl_lab.data.sources import AlpacaDataLoader

loader = AlpacaDataLoader()

# Check if a source supports live data
if isinstance(loader, LiveDataCapable):
    quote = loader.get_latest_quote("AAPL")

# Get all supported features
features = loader.supported_features()
# Returns: {'historical': True, 'live': True, 'streaming': True, ...}
```

### Protocol Definitions

**HistoricalDataCapable:**
```python
def get_historical_ohlcv_data(
    symbols: Union[str, List[str]],
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
    timeframe: str = "1d",
    **kwargs: Any,
) -> pd.DataFrame
```

**LiveDataCapable:**
```python
def get_latest_quote(symbol: str) -> Dict[str, Any]
def get_latest_trade(symbol: str) -> Dict[str, Any]
```

**NewsDataCapable:**
```python
def get_news_data(
    symbols: Union[str, List[str]],
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
    limit: int = 50,
    **kwargs: Any,
) -> pd.DataFrame
```

**FundamentalDataCapable:**
```python
def get_fundamental_data(
    symbol: str,
    metrics: List[str],
    **kwargs: Any,
) -> Dict[str, Any]
```

**MacroDataCapable:**
```python
def get_macro_data(
    indicators: Union[List[str], Dict[str, Dict]],
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    **kwargs: Any,
) -> Dict[str, Any]
```

**AnalystDataCapable:**
```python
def get_historical_grades(symbol: str) -> pd.DataFrame
def get_historical_rating(symbol: str, limit: int = 100) -> pd.DataFrame
```

---

## Data Source Details

### Alpaca

**Best for:** Production trading, real-time data, streaming applications

#### Overview
- **Provider:** Alpaca Markets (https://alpaca.markets)
- **API Documentation:** https://docs.alpaca.markets
- **Implementation:** Uses native Alpaca Python SDK (`alpaca-py`)
- **Protocols:** `HistoricalDataCapable`, `LiveDataCapable`, `StreamingCapable`, `NewsDataCapable`, `ConnectionManaged`

#### API Keys Required
```bash
# .env file
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# Or use: https://api.alpaca.markets for live trading
```

#### Features

**Historical Data:**
- Daily, hourly, minute-level OHLCV data
- Supported timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1w`, `1M`
- Multi-symbol support in single request
- Returns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap

**Real-time Data:**
- Latest quotes (bid/ask prices)
- Latest trades (execution prices)
- Low latency

**Streaming:**
- WebSocket-based real-time updates
- Subscribe to trades, quotes, or bars
- Singleton pattern for connection pooling
- Automatic reconnection

**News:**
- News articles with pagination
- Optional full content inclusion
- Metadata: headline, author, created_at, updated_at, summary, url, images, symbols

#### Usage Examples

```python
from quantrl_lab.data.sources import AlpacaDataLoader

# Initialize
loader = AlpacaDataLoader()

# Historical OHLCV
df = loader.get_historical_ohlcv_data(
    symbols=["AAPL", "GOOGL"],
    start="2024-01-01",
    end="2024-03-01",
    timeframe="1d"
)

# Latest quote
quote = loader.get_latest_quote("AAPL")
print(quote)  # {'symbol': 'AAPL', 'bid_price': 150.25, 'ask_price': 150.30, ...}

# Latest trade
trade = loader.get_latest_trade("AAPL")
print(trade)  # {'symbol': 'AAPL', 'price': 150.27, 'size': 100, 'timestamp': ...}

# News data
news_df = loader.get_news_data(
    symbols="AAPL",
    start="2024-01-01",
    end="2024-01-15",
    limit=10,
    include_content=False
)

# Streaming (WebSocket)
loader.subscribe_to_updates("AAPL", data_type="trades")
loader.start_streaming()
# ... handle updates ...
loader.stop_streaming()
```

#### Limitations
- Requires API key (free tier available)
- Rate limits depend on subscription tier
- Streaming requires persistent WebSocket connection

#### Best Practices
- Use connection pooling (automatically handled via singleton pattern)
- Close connections when done: `loader.disconnect()`
- For backtesting, prefer batch historical requests over streaming
- Check `is_connected()` before making API calls

---

### Yahoo Finance (YFinance)

**Best for:** Quick prototyping, fundamental analysis, free historical data

#### Overview
- **Provider:** Yahoo Finance (via `yfinance` library)
- **API Documentation:** https://github.com/ranaroussi/yfinance
- **Implementation:** Wrapper around `yfinance` library
- **Protocols:** `HistoricalDataCapable`, `FundamentalDataCapable`

#### API Keys Required
None - completely free

#### Features

**Historical Data:**
- Daily, weekly, monthly, and intraday OHLCV
- Supported timeframes: `1m`, `5m`, `15m`, `30m`, `60m`, `1d`, `1wk`, `1mo`
- Multi-symbol support
- Adjusted close prices included
- **Limitation:** 1-minute data limited to last 30 days

**Fundamental Data:**
- Income statements (annual and quarterly)
- Balance sheets (annual and quarterly)
- Cash flow statements (annual and quarterly)
- Combined into single dictionary response

#### Usage Examples

```python
from quantrl_lab.data.sources import YFinanceDataLoader

# Initialize
loader = YFinanceDataLoader()

# Historical OHLCV
df = loader.get_historical_ohlcv_data(
    symbols=["AAPL", "MSFT"],
    start="2024-01-01",
    end="2024-03-01",
    timeframe="1d"
)

# Intraday data (last 30 days only for 1m)
df_intraday = loader.get_historical_ohlcv_data(
    symbols="AAPL",
    start="2024-02-01",
    end="2024-02-07",
    timeframe="1m"
)

# Fundamental data
fundamentals = loader.get_fundamental_data(
    symbol="AAPL",
    frequency="quarterly"  # or "annual"
)

# Access components
income_statement = fundamentals.get("income_statement")
balance_sheet = fundamentals.get("balance_sheet")
cash_flow = fundamentals.get("cash_flow")
```

#### Limitations
- No real-time data
- No streaming
- 1-minute bars limited to 30-day window
- Rate limiting (informal, no official limits but can be throttled)
- Data quality can vary (community-maintained)

#### Best Practices
- Use built-in retry logic (configurable via `max_retries` parameter)
- Avoid excessive requests in short time periods
- For intraday 1m data, request only last 7-30 days
- Cache results locally when possible

---

### Alpha Vantage

**Best for:** Fundamental analysis, macroeconomic research, news sentiment

#### Overview
- **Provider:** Alpha Vantage (https://www.alphavantage.co)
- **API Documentation:** https://www.alphavantage.co/documentation/
- **Implementation:** Direct HTTP requests (mix of SDK and custom)
- **Protocols:** `HistoricalDataCapable`, `FundamentalDataCapable`, `MacroDataCapable`, `NewsDataCapable`

#### API Keys Required
```bash
# .env file
ALPHA_VANTAGE_API_KEY=your_key_here
```

Get free key at: https://www.alphavantage.co/support/#api-key

#### Features

**Historical Data:**
- Daily and intraday OHLCV
- Adjusted and unadjusted prices
- Supported timeframes: `1min`, `5min`, `15min`, `30min`, `60min`, `1d`
- **Free tier:** Last 100 days (`outputsize=compact`)
- **Premium:** 20+ years (`outputsize=full`)

**Fundamental Data:**
- Company overview (sector, industry, market cap, PE ratio, etc.)
- Income statements, balance sheets, cash flow statements
- Earnings data (quarterly and annual)
- Dividends and stock splits

**Macroeconomic Indicators:**
- Real GDP, GDP per capita
- Treasury yields (3-month, 2-year, 5-year, 7-year, 10-year, 30-year)
- Federal funds rate
- CPI (Consumer Price Index)
- Inflation, inflation expectation
- Consumer sentiment
- Retail sales
- Unemployment rate
- Nonfarm payroll

**News Sentiment:**
- News articles with sentiment analysis
- Ticker-specific sentiment scores
- Relevance scores
- Source tracking

#### Usage Examples

```python
from quantrl_lab.data.sources import AlphaVantageDataLoader
from quantrl_lab.utils.config import FundamentalMetric, MacroIndicator

# Initialize
loader = AlphaVantageDataLoader()

# Historical OHLCV (daily, free tier)
df = loader.get_historical_ohlcv_data(
    symbols="AAPL",
    timeframe="1d"
    # Note: Don't specify start/end for free tier (gets last 100 days)
)

# Intraday data (PREMIUM REQUIRED for historical months)
df_intraday = loader.get_historical_ohlcv_data(
    symbols="AAPL",
    timeframe="5min"
    # month="2024-01"  # Requires premium
)

# Company overview
fundamentals = loader.get_fundamental_data(
    symbol="AAPL",
    metrics=[FundamentalMetric.COMPANY_OVERVIEW]
)
overview = fundamentals["company_overview"]

# Financial statements
financials = loader.get_fundamental_data(
    symbol="MSFT",
    metrics=[
        FundamentalMetric.INCOME_STATEMENT,
        FundamentalMetric.BALANCE_SHEET,
        FundamentalMetric.CASH_FLOW,
    ]
)

# Macroeconomic data
macro_data = loader.get_macro_data(
    indicators=[MacroIndicator.REAL_GDP, MacroIndicator.CPI],
    start="2020-01-01",
    end="2024-01-01"
)

# Treasury yields with custom parameters
treasury = loader.get_macro_data(
    indicators={
        MacroIndicator.TREASURY_YIELD: {
            "interval": "monthly",
            "maturity": "10year"
        }
    },
    start="2023-01-01",
    end="2024-01-01"
)

# News with sentiment
news_df = loader.get_news_data(
    symbols="AAPL",
    start="2024-01-01",
    end="2024-01-15",
    limit=10
)
```

#### Free Tier Limitations
- **25 API requests per day** (not per minute - strict daily quota)
- **1 request per second** burst limit
- **`outputsize=compact` only** (last 100 data points for daily, ~2 weeks for intraday)
- **No historical intraday with `month` parameter** (requires premium)
- **No adjusted intraday data** (requires premium)

The loader automatically:
- Enforces rate limiting (1.2s between requests)
- Defaults to `outputsize=compact`
- Handles API response errors gracefully

#### Best Practices
- **Cache aggressively** - with only 25 requests/day, minimize API calls
- Use `outputsize='compact'` explicitly on free tier
- Batch requests when possible (but respect 1/sec limit)
- For macro data, request less frequent intervals (monthly vs daily)
- Monitor daily quota usage
- Consider premium plan for production use or historical intraday data

---

### Financial Modeling Prep (FMP)

**Best for:** Intraday data, analyst sentiment, alternative to Alpha Vantage

#### Overview
- **Provider:** Financial Modeling Prep (https://financialmodelingprep.com)
- **API Documentation:** https://site.financialmodelingprep.com/developer/docs
- **Implementation:** HTTP requests via `HTTPRequestWrapper` with retry logic
- **Protocols:** `HistoricalDataCapable`, `AnalystDataCapable`

#### API Keys Required
```bash
# .env file
FMP_API_KEY=your_key_here
```

Get free key at: https://financialmodelingprep.com/developer/docs/

#### Features

**Historical Data:**
- Daily and intraday OHLCV
- Supported intraday timeframes: `5min`, `15min`, `30min`, `1hour`, `4hour`
- Daily timeframe: `1d`
- Standardized DataFrame output (symbol, timestamp, open, high, low, close, volume)
- **Limitation:** Single symbol per request (logs warning if multiple provided)

**Analyst Data (NEW):**
- Historical analyst grades/recommendations
- Historical analyst ratings with configurable limit
- Unique to FMP among available sources

#### Usage Examples

```python
from quantrl_lab.data.sources import FMPDataSource

# Initialize
loader = FMPDataSource()

# Historical daily data
df = loader.get_historical_ohlcv_data(
    symbols="AAPL",
    start="2024-01-01",
    end="2024-06-01",
    timeframe="1d"
)

# Intraday data (5-minute bars)
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=7)

df_intraday = loader.get_historical_ohlcv_data(
    symbols="AAPL",
    start=start_date,
    end=end_date,
    timeframe="5min"
)

# Different intraday timeframes
timeframes = ["15min", "1hour", "4hour"]
for tf in timeframes:
    df_tf = loader.get_historical_ohlcv_data(
        symbols="MSFT",
        start=start_date,
        end=end_date,
        timeframe=tf
    )

# Analyst grades
grades = loader.get_historical_grades("AAPL")
print(grades.columns.tolist())

# Analyst ratings
ratings = loader.get_historical_rating("AAPL", limit=50)
print(ratings.head())
```

#### Limitations
- **Single symbol per request** (multi-symbol lists will use only first symbol with warning)
- No fundamental data (use YFinance or Alpha Vantage instead)
- No news data
- No real-time/streaming capabilities
- Free tier limitations vary by endpoint

#### Best Practices
- Use single symbol strings, not lists
- Good alternative for intraday data if Alpha Vantage premium not available
- Leverage analyst data for sentiment analysis features
- Built-in retry logic handles transient failures

---

## Data Utilities

### Overview

As of commit 09d7695, all data loaders use centralized utility modules from `src/quantrl_lab/data/utils/`. This refactoring eliminated code duplication and improved consistency.

### Available Utilities

#### 1. Date Parsing (`date_parsing.py`)

**Functions:**
```python
normalize_date(date_input: Union[str, datetime, date]) -> datetime
normalize_date_range(
    start: Union[str, datetime, date],
    end: Optional[Union[str, datetime, date]] = None,
    default_end_to_now: bool = True,
    validate_order: bool = True
) -> Tuple[datetime, datetime]
format_date_to_string(date_obj: datetime, fmt: str = "%Y-%m-%d") -> str
```

**Usage:**
```python
from quantrl_lab.data.utils.date_parsing import normalize_date, normalize_date_range

# Convert various formats to datetime
dt = normalize_date("2024-01-01")  # str to datetime
dt = normalize_date(datetime(2024, 1, 1))  # pass-through

# Get validated date range
start, end = normalize_date_range(
    start="2024-01-01",
    end="2024-03-01",
    validate_order=True
)
```

#### 2. Symbol Handling (`symbol_handling.py`)

**Functions:**
```python
normalize_symbols(symbols: Union[str, List[str]]) -> List[str]
validate_symbols(symbols: Union[str, List[str]]) -> List[str]
get_single_symbol(
    symbols: Union[str, List[str]],
    warn_on_multiple: bool = True
) -> str
```

**Usage:**
```python
from quantrl_lab.data.utils.symbol_handling import normalize_symbols, get_single_symbol

# Normalize to list
symbols = normalize_symbols("AAPL")  # Returns: ["AAPL"]
symbols = normalize_symbols(["AAPL", "GOOGL"])  # Returns: ["AAPL", "GOOGL"]

# Extract first symbol (useful for FMP)
symbol = get_single_symbol(["AAPL", "GOOGL"], warn_on_multiple=True)
# Logs warning: "Multiple symbols provided, using first: AAPL"
```

#### 3. DataFrame Normalization (`dataframe_normalization.py`)

**Functions:**
```python
standardize_ohlcv_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame
standardize_ohlcv_dataframe(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
    symbol: Optional[str] = None,
    timestamp_col: str = "timestamp",
    add_date: bool = True,
    sort_data: bool = True,
    convert_numeric: bool = True
) -> pd.DataFrame
add_date_column_from_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame
add_symbol_column(df: pd.DataFrame, symbol: str) -> pd.DataFrame
sort_by_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame
convert_columns_to_numeric(df: pd.DataFrame, columns: List[str], errors: str = "coerce") -> pd.DataFrame
```

**Usage:**
```python
from quantrl_lab.data.utils.dataframe_normalization import standardize_ohlcv_dataframe

# Comprehensive standardization pipeline
df_standardized = standardize_ohlcv_dataframe(
    df=raw_df,
    column_mapping={"Open": "open", "High": "high", ...},
    symbol="AAPL",
    add_date=True,
    sort_data=True,
    convert_numeric=True
)
```

#### 4. Response Validation (`response_validation.py`)

**Functions:**
```python
validate_api_response(response_data: Any) -> None
convert_to_dataframe_safe(
    data: Any,
    expected_min_rows: int = 1,
    symbol: Optional[str] = None
) -> pd.DataFrame
check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None
log_dataframe_info(df: pd.DataFrame, message: str, symbol: Optional[str] = None) -> None
validate_date_range_data(
    df: pd.DataFrame,
    date_col: str,
    expected_date_range: Optional[Tuple[datetime, datetime]] = None
) -> None
```

**Usage:**
```python
from quantrl_lab.data.utils.response_validation import validate_api_response, convert_to_dataframe_safe

# Validate API response
validate_api_response(api_response)  # Raises ValueError if invalid

# Safe DataFrame conversion with validation
df = convert_to_dataframe_safe(
    data=api_response,
    expected_min_rows=1,
    symbol="AAPL"
)
```

#### 5. HTTP Request Wrapper (`request_utils.py`)

**Classes and Enums:**
```python
class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"

class HTTPRequestWrapper:
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        rate_limit_delay: Optional[float] = None
    )

    def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Response
    def post(self, url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, headers: Optional[Dict] = None) -> requests.Response

def create_default_wrapper() -> HTTPRequestWrapper
```

**Usage:**
```python
from quantrl_lab.data.utils.request_utils import HTTPRequestWrapper, RetryStrategy

# Create wrapper with exponential backoff
wrapper = HTTPRequestWrapper(
    max_retries=3,
    retry_delay=1.0,
    retry_strategy=RetryStrategy.EXPONENTIAL,
    rate_limit_delay=1.2  # For Alpha Vantage
)

# Make request (automatic retries on failure)
response = wrapper.get("https://api.example.com/data", params={"symbol": "AAPL"})
data = response.json()
```

### Benefits of Centralized Utilities

1. **Consistency**: All data loaders use identical validation and normalization logic
2. **Maintainability**: Fix bugs in one place, benefits all loaders
3. **Testability**: 118 tests cover utility functions comprehensively
4. **Reusability**: Easy to extend utilities for custom data sources

---

## Best Practices

### 1. Choosing the Right Data Source

**For backtesting:**
- Daily data: YFinance (free) or Alpaca (reliable)
- Intraday data: FMP or Alpaca (avoid Alpha Vantage free tier)
- Long historical windows: Alpaca or YFinance

**For production:**
- Real-time trading: Alpaca only (streaming, live quotes)
- Fundamental analysis: YFinance or Alpha Vantage
- Macroeconomic research: Alpha Vantage only

**For feature engineering:**
- Analyst sentiment: FMP (grades/ratings)
- News sentiment: Alpha Vantage or Alpaca
- Economic indicators: Alpha Vantage (Fed data, GDP, CPI)

### 2. API Key Management

```bash
# .env file (NEVER commit to version control)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPHA_VANTAGE_API_KEY=your_key
FMP_API_KEY=your_key
```

```python
# Load from environment
from dotenv import load_dotenv
load_dotenv()

# Initialize loaders (automatically reads .env)
loader = AlpacaDataLoader()
```

### 3. Error Handling

All loaders implement comprehensive error handling:

```python
try:
    df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start="2024-01-01",
        end="2024-03-01"
    )
except ValueError as e:
    print(f"Invalid parameters: {e}")
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 4. Caching Results

For expensive API calls (especially Alpha Vantage with daily quota):

```python
import pandas as pd
from pathlib import Path

cache_dir = Path("data_cache")
cache_dir.mkdir(exist_ok=True)

cache_file = cache_dir / "aapl_daily_2024.parquet"

if cache_file.exists():
    df = pd.read_parquet(cache_file)
else:
    df = loader.get_historical_ohlcv_data(...)
    df.to_parquet(cache_file)
```

### 5. Rate Limiting

**Alpha Vantage (automatic):**
```python
# Loader automatically enforces 1.2s delay between requests
loader = AlphaVantageDataLoader()
```

**Custom rate limiting:**
```python
import time

for symbol in symbols:
    df = loader.get_historical_ohlcv_data(symbols=symbol, ...)
    time.sleep(1.0)  # Manual rate limiting if needed
```

### 6. Multi-Symbol Requests

**Efficient (single request):**
```python
# Alpaca, YFinance, Alpha Vantage support multi-symbol
df = loader.get_historical_ohlcv_data(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start="2024-01-01",
    end="2024-03-01"
)
```

**Inefficient (multiple requests):**
```python
# Avoid this for sources with multi-symbol support
dfs = []
for symbol in ["AAPL", "GOOGL", "MSFT"]:
    df = loader.get_historical_ohlcv_data(symbols=symbol, ...)
    dfs.append(df)
df = pd.concat(dfs)
```

**FMP (single symbol only):**
```python
# FMP requires one request per symbol
for symbol in ["AAPL", "GOOGL"]:
    df = loader.get_historical_ohlcv_data(symbols=symbol, ...)
```

### 7. Date Handling

All loaders accept flexible date formats:

```python
from datetime import datetime, timedelta

# String format
df = loader.get_historical_ohlcv_data(symbols="AAPL", start="2024-01-01", end="2024-03-01")

# Datetime objects
start = datetime(2024, 1, 1)
end = datetime(2024, 3, 1)
df = loader.get_historical_ohlcv_data(symbols="AAPL", start=start, end=end)

# Relative dates
end = datetime.now()
start = end - timedelta(days=30)
df = loader.get_historical_ohlcv_data(symbols="AAPL", start=start, end=end)
```

### 8. Logging

Enable debug logging to troubleshoot issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("quantrl_lab")
logger.setLevel(logging.DEBUG)

# Now loader will output detailed logs
df = loader.get_historical_ohlcv_data(...)
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Symptoms:**
```
ERROR: Alpha Vantage API key not configured.
```

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify key is set
cat .env | grep ALPHA_VANTAGE_API_KEY

# Reload environment
from dotenv import load_dotenv
load_dotenv(override=True)
```

#### 2. Rate Limit Exceeded (Alpha Vantage)

**Symptoms:**
```
API rate limit exceeded. Please wait and try again.
```

**Solution:**
- Wait 24 hours for quota reset (free tier: 25 req/day)
- Use cached data
- Upgrade to premium plan
- Switch to YFinance or FMP for historical data

#### 3. Empty DataFrame Returned

**Possible causes:**
- Invalid symbol
- Date range outside available data
- API endpoint returned no data

**Debugging:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

df = loader.get_historical_ohlcv_data(...)

if df.empty:
    print("No data returned. Check:")
    print("1. Symbol is valid")
    print("2. Date range has data")
    print("3. API key is valid")
    print("4. Loader debug logs above")
```

#### 4. YFinance 1m Data Limited

**Symptoms:**
```
Warning: 1m data limited to last 30 days
```

**Solution:**
- Reduce date range to last 30 days for 1m timeframe
- Use 5m, 15m, or higher timeframes for longer periods
- Switch to Alpaca or FMP for historical intraday

#### 5. FMP Multi-Symbol Warning

**Symptoms:**
```
WARNING: Multiple symbols provided, using first: AAPL
```

**Solution:**
```python
# Loop through symbols for FMP
for symbol in ["AAPL", "GOOGL", "MSFT"]:
    df = fmp_loader.get_historical_ohlcv_data(symbols=symbol, ...)
```

#### 6. Connection Timeout

**Symptoms:**
```
requests.exceptions.Timeout: Request timed out
```

**Solution:**
```python
# Increase timeout in HTTPRequestWrapper
from quantrl_lab.data.utils.request_utils import HTTPRequestWrapper

wrapper = HTTPRequestWrapper(timeout=60)  # Default: 30s
```

#### 7. SSL Certificate Errors

**Symptoms:**
```
requests.exceptions.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution:**
```python
# Update certifi package
pip install --upgrade certifi

# Or use environment variable (not recommended for production)
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'
```

---

## Migration Guide

### Updating from Pre-Refactor Code

If you have custom data loaders or code written before commit 09d7695, update to use new utilities:

**Before (duplicate logic):**
```python
# Old approach: inline date parsing
from datetime import datetime

if isinstance(start, str):
    start = datetime.strptime(start, "%Y-%m-%d")
if isinstance(end, str):
    end = datetime.strptime(end, "%Y-%m-%d")
```

**After (centralized utility):**
```python
# New approach: use utility function
from quantrl_lab.data.utils.date_parsing import normalize_date_range

start, end = normalize_date_range(start, end, validate_order=True)
```

**Before (manual DataFrame normalization):**
```python
# Old: manual column renaming
df.rename(columns={"Open": "open", "High": "high", ...}, inplace=True)
df["timestamp"] = pd.to_datetime(df.index)
df = df.sort_values("timestamp")
```

**After (standardized pipeline):**
```python
# New: one-line standardization
from quantrl_lab.data.utils.dataframe_normalization import standardize_ohlcv_dataframe

df = standardize_ohlcv_dataframe(df, column_mapping={...}, add_date=True, sort_data=True)
```

---

## Further Reading

- [AGENTS.md](../AGENTS.md) - Developer guide with project structure
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design patterns
- [examples/](../examples/) - Example scripts for each data source
- [API Reference](https://whanyu1212.github.io/QuantRL-Lab/) - Full API documentation

---

**Last Updated:** 2026-01-30
**Recent Changes:** Added FMP data source, refactored utilities (commit 09d7695)

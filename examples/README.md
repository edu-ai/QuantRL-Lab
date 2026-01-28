# Data Source Examples

This folder contains example scripts demonstrating how to fetch data from different sources in QuantRL-Lab.

## Available Examples

| Script | Data Source | API Key Required |
|--------|-------------|------------------|
| `fetch_yfinance_data.py` | Yahoo Finance | No |
| `fetch_alpaca_data.py` | Alpaca | Yes |
| `fetch_alphavantage_data.py` | Alpha Vantage | Yes |
| `fetch_fmp_data.py` | Financial Modeling Prep | Yes |

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure API keys (for Alpaca, Alpha Vantage, and FMP):
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Running Examples

```bash
# Yahoo Finance (no API key needed)
uv run python examples/fetch_yfinance_data.py

# Alpaca (requires API key)
uv run python examples/fetch_alpaca_data.py

# Alpha Vantage (requires API key)
uv run python examples/fetch_alphavantage_data.py

# Financial Modeling Prep (requires API key)
uv run python examples/fetch_fmp_data.py
```

## Data Source Comparison

### Yahoo Finance
- **Pros**: Free, no API key, includes fundamental data
- **Cons**: Rate limits on intraday data (30 days max for 1-minute bars)
- **Best for**: Quick prototyping, backtesting with daily data

### Alpaca
- **Pros**: Real-time quotes/trades, news data, reliable historical data
- **Cons**: Requires API key (free tier available)
- **Best for**: Production systems, real-time applications

### Alpha Vantage
- **Pros**: Extensive fundamental data, macroeconomic indicators, news sentiment
- **Cons**: Strict rate limits on free tier (25 calls/day, 1 req/sec); intraday data and `outputsize=full` require premium
- **Best for**: Fundamental analysis, macro research (premium recommended for historical OHLCV)

### Financial Modeling Prep (FMP)
- **Pros**: Good intraday data coverage (5min, 15min, 30min, 1hour, 4hour), analyst grades/ratings
- **Cons**: Requires API key, single symbol per request
- **Best for**: Intraday analysis, analyst sentiment data

## Data Capabilities by Source

| Capability | Yahoo Finance | Alpaca | Alpha Vantage |
|------------|---------------|--------|---------------|
| Historical OHLCV | Yes | Yes | Yes (last 100 days free) |
| Intraday Data | Limited | Yes | Premium only |
| Fundamental Data | Yes | No | Yes |
| News Data | No | Yes | Yes |
| Macro Indicators | No | No | Yes |
| Real-time Quotes | No | Yes | No |
| Streaming | No | Yes* | No |

*Streaming is available but excluded from these examples.

## Alpha Vantage Free Tier Limitations

The free tier has significant restrictions:
- **25 requests/day** (not per minute)
- **1 request/second** burst limit
- **`outputsize=full`** (20+ years of data) requires premium
- **Intraday data** (1min, 5min, etc.) requires premium
- **Historical intraday with `month` parameter** requires premium

The `AlphaVantageDataLoader` automatically handles rate limiting (1.2s between requests) and defaults to `outputsize=compact` (last 100 data points) for free tier compatibility.

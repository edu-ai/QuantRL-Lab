# Data Sources Guide

QuantRL-Lab supports four financial data providers through a unified protocol-based interface. All sources return normalized pandas DataFrames with consistent column names (`symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`).

For the protocol architecture that underpins this, see [Architecture — Protocol Pattern](ARCHITECTURE.md#protocol-pattern-for-data-sources).

## Table of Contents
- [Capability Matrix](#capability-matrix)
- [Use Case Recommendations](#use-case-recommendations)
- [Alpaca](#alpaca)
- [Yahoo Finance](#yahoo-finance)
- [Alpha Vantage](#alpha-vantage)
- [Financial Modeling Prep (FMP)](#financial-modeling-prep-fmp)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Capability Matrix

| Capability | Alpaca | YFinance | Alpha Vantage | FMP |
|------------|:------:|:--------:|:-------------:|:---:|
| Historical OHLCV (daily) | ✅ | ✅ | ✅ (100 days free) | ✅ |
| Intraday data | ✅ | ✅ (30-day limit for 1m) | 🔒 Premium | ✅ |
| Real-time quotes/trades | ✅ | ❌ | ❌ | ❌ |
| Streaming (WebSocket) | ✅ | ❌ | ❌ | ❌ |
| Fundamental data | ❌ | ✅ | ✅ | ❌ |
| Macroeconomic indicators | ❌ | ❌ | ✅ | ❌ |
| News data | ✅ | ❌ | ✅ | ❌ |
| Analyst grades/ratings | ❌ | ❌ | ❌ | ✅ |
| Sector/industry performance | ❌ | ❌ | ❌ | ✅ |
| Company profile | ❌ | ❌ | ✅ | ✅ |
| Multi-symbol in one request | ✅ | ✅ | ✅ | ⚠️ First only |
| API key required | ✅ | ❌ | ✅ | ✅ |

---

## Use Case Recommendations

| Use Case | Recommended Source |
|----------|--------------------|
| Quick prototyping / backtesting | YFinance — free, no setup |
| Production / real-time trading | Alpaca — streaming, live quotes |
| Fundamental / macro research | Alpha Vantage — financials + GDP/CPI/yields |
| Analyst sentiment features | FMP — historical grades and ratings |
| Intraday backtesting | Alpaca or FMP |
| News sentiment | Alpaca or Alpha Vantage |

---

## Alpaca

**Protocols:** `HistoricalDataCapable`, `LiveDataCapable`, `StreamingCapable`, `NewsDataCapable`, `ConnectionManaged`

**API keys** (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY` in `.env`): [alpaca.markets](https://alpaca.markets)

### Features

- Historical OHLCV: `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1w`, `1M`
- Real-time: latest quotes (bid/ask) and trades
- Streaming: WebSocket subscription to trades, quotes, or bars with automatic reconnection
- News: articles with headline, summary, symbols, sentiment metadata

### Usage

```python
from quantrl_lab.data.sources import AlpacaDataLoader

loader = AlpacaDataLoader()

# Historical OHLCV
df = loader.get_historical_ohlcv_data(
    symbols=["AAPL", "GOOGL"],
    start="2024-01-01",
    end="2024-03-01",
    timeframe="1d"
)

# Real-time
quote = loader.get_latest_quote("AAPL")
trade = loader.get_latest_trade("AAPL")

# News
news_df = loader.get_news_data(symbols="AAPL", start="2024-01-01", end="2024-01-15")

# Streaming
loader.subscribe_to_updates("AAPL", data_type="trades")
await loader.start_streaming()
await loader.stop_streaming()
```

### Limitations
- Requires API key (free tier available)
- Streaming requires a persistent WebSocket connection — call `loader.disconnect()` when done

---

## Yahoo Finance

**Protocols:** `HistoricalDataCapable`, `FundamentalDataCapable`

**API key:** None — completely free.

### Features

- Historical OHLCV: `1m`, `5m`, `15m`, `30m`, `60m`, `1d`, `1wk`, `1mo`
- Fundamental data: income statements, balance sheets, cash flow (annual and quarterly)
- Adjusted close prices included

### Usage

```python
from quantrl_lab.data.sources import YFinanceDataLoader

loader = YFinanceDataLoader()

# Historical OHLCV
df = loader.get_historical_ohlcv_data(
    symbols=["AAPL", "MSFT"],
    start="2024-01-01",
    end="2024-03-01",
    timeframe="1d"
)

# Fundamental data
fundamentals = loader.get_fundamental_data(symbol="AAPL", frequency="quarterly")
income_statement = fundamentals.get("income_statement")
balance_sheet = fundamentals.get("balance_sheet")
cash_flow = fundamentals.get("cash_flow")
```

### Limitations
- No real-time or streaming data
- 1-minute bars limited to last 30 days; use `5m`+ for longer windows
- Informal rate limiting — avoid hammering the API in tight loops

---

## Alpha Vantage

**Protocols:** `HistoricalDataCapable`, `FundamentalDataCapable`, `MacroDataCapable`, `NewsDataCapable`

**API key** (`ALPHA_VANTAGE_API_KEY` in `.env`): [alphavantage.co](https://www.alphavantage.co/support/#api-key)

### Features

- Historical OHLCV: `1min`, `5min`, `15min`, `30min`, `60min`, `1d`; free tier gives last 100 data points
- Fundamental data: company overview, income statements, balance sheets, cash flow, earnings, dividends
- Macroeconomic indicators: real GDP, CPI, Fed funds rate, treasury yields (3m–30yr), unemployment, nonfarm payroll, retail sales, consumer sentiment
- News: articles with per-ticker sentiment scores and relevance scores

### Usage

```python
from quantrl_lab.data.sources import AlphaVantageDataLoader
from quantrl_lab.data.config import FundamentalMetric, MacroIndicator

loader = AlphaVantageDataLoader()

# Historical OHLCV (free tier: omit start/end to get last 100 days)
df = loader.get_historical_ohlcv_data(symbols="AAPL", timeframe="1d")

# Fundamental data
fundamentals = loader.get_fundamental_data(
    symbol="AAPL",
    metrics=[FundamentalMetric.INCOME_STATEMENT, FundamentalMetric.BALANCE_SHEET]
)

# Macro data
macro = loader.get_macro_data(
    indicators=[MacroIndicator.REAL_GDP, MacroIndicator.CPI],
    start="2020-01-01",
    end="2024-01-01"
)

# Treasury yield with parameters
treasury = loader.get_macro_data(
    indicators={MacroIndicator.TREASURY_YIELD: {"interval": "monthly", "maturity": "10year"}},
    start="2023-01-01",
    end="2024-01-01"
)

# News with sentiment
news_df = loader.get_news_data(symbols="AAPL", start="2024-01-01", end="2024-01-15")
```

### Limitations
- **Free tier: 25 requests/day, 1 request/second** — the loader enforces 1.2s between requests automatically
- Free tier caps OHLCV at last 100 data points (`outputsize=compact`); full history and intraday `month` parameter require premium
- Cache aggressively — with 25 req/day, every call counts

---

## Financial Modeling Prep (FMP)

**Protocols:** `HistoricalDataCapable`, `AnalystDataCapable`, `SectorDataCapable`, `CompanyProfileCapable`

**API key** (`FMP_API_KEY` in `.env`): [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs/)

### Features

- Historical OHLCV: `1d`, `5min`, `15min`, `30min`, `1hour`, `4hour`
- Analyst data: historical grades/recommendations and ratings — unique among available sources
- Sector/industry performance: historical performance for any sector or industry (useful for market context features)
- Company profile: sector, industry, market cap, beta, CEO, exchange, IPO date

### Usage

```python
from quantrl_lab.data.sources import FMPDataSource

loader = FMPDataSource()

# Historical OHLCV (single symbol only)
df = loader.get_historical_ohlcv_data(symbols="AAPL", start="2024-01-01", end="2024-06-01", timeframe="1d")

# Intraday
df_intraday = loader.get_historical_ohlcv_data(symbols="AAPL", start="2024-02-01", end="2024-02-07", timeframe="5min")

# Analyst data
grades = loader.get_historical_grades("AAPL")
ratings = loader.get_historical_rating("AAPL", limit=50)

# Sector/industry performance
sector_perf = loader.get_historical_sector_performance("Technology")
industry_perf = loader.get_historical_industry_performance("Biotechnology")

# Company profile
profile = loader.get_company_profile("AAPL")
```

### Limitations
- **Single symbol per request** — passing a list uses only the first symbol (with a warning)
- No fundamental data, news, or real-time capabilities

---

## Best Practices

**Choosing a source:**
- Daily backtesting: YFinance (free) or Alpaca (reliable, consistent)
- Intraday backtesting: Alpaca or FMP
- Macro/fundamental research: Alpha Vantage
- Analyst sentiment features: FMP
- Production real-time: Alpaca only

**API key management:**
```bash
# .env (never commit)
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPHA_VANTAGE_API_KEY=...
FMP_API_KEY=...
```
All loaders read keys from environment variables automatically via `python-dotenv`.

**Caching:** For sources with strict quotas (especially Alpha Vantage), save fetched data locally:
```python
cache_file = Path("cache/aapl_daily.parquet")
if cache_file.exists():
    df = pd.read_parquet(cache_file)
else:
    df = loader.get_historical_ohlcv_data(...)
    df.to_parquet(cache_file)
```

**Multi-symbol requests:** Alpaca, YFinance, and Alpha Vantage support lists in a single call. FMP requires one request per symbol — loop manually.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `API key not configured` | Key missing from environment | Check `.env` file and call `load_dotenv()` |
| `Rate limit exceeded` (Alpha Vantage) | Hit 25 req/day free tier | Wait 24h, use cached data, or upgrade |
| Empty DataFrame returned | Bad symbol, out-of-range dates, or API error | Enable `logging.DEBUG` to see the raw API response |
| `1m data limited to last 30 days` (YFinance) | yfinance API restriction | Use `5m`+ for longer periods, or switch to Alpaca/FMP |
| `Multiple symbols provided, using first` (FMP) | FMP single-symbol limitation | Loop through symbols individually |

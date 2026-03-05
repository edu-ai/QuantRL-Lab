# New Protocol Reference Card

## Quick Reference for SectorDataCapable and CompanyProfileCapable

This guide provides a quick reference for the newly added protocols in QuantRL-Lab.

---

## SectorDataCapable Protocol

### Purpose
Access historical performance data for market sectors and industries.

### Protocol Interface
```python
from quantrl_lab.data.interface import SectorDataCapable

@runtime_checkable
class SectorDataCapable(Protocol):
    def get_historical_sector_performance(
        self,
        sector: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Get historical performance data for a specific market sector."""
        ...

    def get_historical_industry_performance(
        self,
        industry: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Get historical performance data for a specific industry."""
        ...
```

### Implementing Data Sources
- **FMPDataSource** ✅

### Example Usage

```python
from quantrl_lab.data.interface import SectorDataCapable
from quantrl_lab.data.sources import FMPDataSource

# Initialize
loader = FMPDataSource()

# Check if supported (runtime check)
if isinstance(loader, SectorDataCapable):
    # Get sector performance
    energy = loader.get_historical_sector_performance("Energy")
    tech = loader.get_historical_sector_performance("Technology")

    # Get industry performance
    software = loader.get_historical_industry_performance("Software")
    biotech = loader.get_historical_industry_performance("Biotechnology")
```

### Common Sectors (FMP)
- Energy
- Technology
- Healthcare
- Financials
- Consumer Cyclical
- Industrials
- Basic Materials
- Consumer Defensive
- Real Estate
- Utilities
- Communication Services

### Common Industries (FMP)
- Software
- Biotechnology
- Banks
- Oil & Gas
- Semiconductors
- Insurance
- Auto Manufacturers
- Pharmaceuticals
- Consumer Electronics
- Aerospace & Defense

---

## CompanyProfileCapable Protocol

### Purpose
Access company profile and metadata including sector, industry, executives, and key metrics.

### Protocol Interface
```python
from quantrl_lab.data.interface import CompanyProfileCapable

@runtime_checkable
class CompanyProfileCapable(Protocol):
    def get_company_profile(
        self,
        symbol: Union[str, List[str]],
        **kwargs: Any
    ) -> pd.DataFrame:
        """Get company profile information including sector, industry, and key metrics."""
        ...
```

### Implementing Data Sources
- **FMPDataSource** ✅

### Example Usage

```python
from quantrl_lab.data.interface import CompanyProfileCapable
from quantrl_lab.data.sources import FMPDataSource

# Initialize
loader = FMPDataSource()

# Check if supported (runtime check)
if isinstance(loader, CompanyProfileCapable):
    # Get company profile
    profile = loader.get_company_profile("AAPL")

    # Access profile data
    company_name = profile.iloc[0]['companyName']
    sector = profile.iloc[0]['sector']
    industry = profile.iloc[0]['industry']
    ceo = profile.iloc[0].get('ceo', 'N/A')
    market_cap = profile.iloc[0].get('mktCap', 0)
    website = profile.iloc[0].get('website', 'N/A')
```

### Available Fields (FMP)

**Basic Information:**
- `symbol` - Stock ticker
- `companyName` - Full company name
- `sector` - Sector classification
- `industry` - Industry classification
- `description` - Business description
- `website` - Company website URL

**Executive Information:**
- `ceo` - Chief Executive Officer name

**Trading Information:**
- `exchange` - Stock exchange
- `exchangeShortName` - Exchange abbreviation
- `currency` - Trading currency
- `ipoDate` - Initial public offering date

**Financial Metrics:**
- `mktCap` - Market capitalization
- `price` - Current stock price
- `beta` - Stock beta
- `volAvg` - Average volume

**Location:**
- `address`, `city`, `state`, `zip`, `country`
- `phone` - Contact phone number

**Other:**
- `fullTimeEmployees` - Number of employees
- `image` - Company logo URL
- `isEtf`, `isActivelyTrading`, `isFund`, `isAdr` - Asset type flags

---

## Feature Detection

### Using isinstance()
```python
from quantrl_lab.data.interface import SectorDataCapable, CompanyProfileCapable
from quantrl_lab.data.sources import FMPDataSource

loader = FMPDataSource()

# Runtime protocol checking
has_sector_data = isinstance(loader, SectorDataCapable)
has_company_profile = isinstance(loader, CompanyProfileCapable)

print(f"Supports sector data: {has_sector_data}")
print(f"Supports company profile: {has_company_profile}")
```

### Using supports_feature()
```python
from quantrl_lab.data.sources import FMPDataSource

loader = FMPDataSource()

# Check specific features
if loader.supports_feature("sector_data"):
    print("This source supports sector/industry performance data")

if loader.supports_feature("company_profile"):
    print("This source supports company profiles")

# Get all supported features
features = loader.supported_features
print(f"All features: {features}")
# Output: ['historical_bars', 'analyst_data', 'sector_data', 'company_profile']
```

---

## Use Cases

### Sector Rotation Analysis
```python
from quantrl_lab.data.sources import FMPDataSource

loader = FMPDataSource()

# Compare multiple sectors
sectors = ["Technology", "Healthcare", "Financials", "Energy"]
sector_performance = {}

for sector in sectors:
    sector_performance[sector] = loader.get_historical_sector_performance(sector)

# Analyze which sectors are outperforming
# ... your analysis code here ...
```

### Stock Screening by Sector/Industry
```python
from quantrl_lab.data.sources import FMPDataSource

loader = FMPDataSource()

# Screen stocks by industry
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
tech_stocks = []

for symbol in symbols:
    profile = loader.get_company_profile(symbol)
    if profile.iloc[0]['sector'] == 'Technology':
        tech_stocks.append(symbol)

print(f"Technology stocks: {tech_stocks}")
```

### Industry Trend Analysis
```python
from quantrl_lab.data.sources import FMPDataSource

loader = FMPDataSource()

# Track specific industry
biotech_perf = loader.get_historical_industry_performance("Biotechnology")

# Analyze trends
# ... your analysis code here ...
```

### Company Metadata Enrichment
```python
from quantrl_lab.data.sources import FMPDataSource
import pandas as pd

loader = FMPDataSource()

# Build dataset with company metadata
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
profiles = []

for symbol in symbols:
    profile = loader.get_company_profile(symbol)
    profiles.append(profile)

# Combine into single DataFrame
company_data = pd.concat(profiles, ignore_index=True)

# Use for analysis or filtering
print(company_data[['symbol', 'companyName', 'sector', 'industry', 'mktCap']])
```

---

## Testing Protocol Support

### Writing Protocol-Aware Code
```python
from typing import Protocol
from quantrl_lab.data.interface import SectorDataCapable, CompanyProfileCapable

def analyze_sector_trends(data_source):
    """Analyze sector trends if the data source supports it."""
    if not isinstance(data_source, SectorDataCapable):
        raise ValueError(f"{data_source.source_name} does not support sector data")

    # Safe to call sector data methods
    energy = data_source.get_historical_sector_performance("Energy")
    tech = data_source.get_historical_sector_performance("Technology")

    # ... analysis logic ...

def enrich_with_company_data(data_source, symbols):
    """Enrich symbols with company profile data if available."""
    if not isinstance(data_source, CompanyProfileCapable):
        print(f"Warning: {data_source.source_name} does not support company profiles")
        return None

    profiles = []
    for symbol in symbols:
        profile = data_source.get_company_profile(symbol)
        profiles.append(profile)

    return pd.concat(profiles, ignore_index=True)
```

---

## Migration Guide

### From Manual Checks to Protocol-Based
**Before:**
```python
# Manual feature checking (fragile)
if hasattr(loader, 'get_historical_sector_performance'):
    sector_data = loader.get_historical_sector_performance("Energy")
```

**After:**
```python
# Protocol-based checking (type-safe)
from quantrl_lab.data.interface import SectorDataCapable

if isinstance(loader, SectorDataCapable):
    sector_data = loader.get_historical_sector_performance("Energy")
```

### Benefits
1. **Type Safety:** IDE autocomplete and type checkers understand protocols
2. **Documentation:** Protocol docstrings document expected behavior
3. **Discoverability:** `supported_features` property lists all capabilities
4. **Consistency:** All data sources follow the same interface pattern

---

## Further Reading

- [DATA_SOURCES.md](DATA_SOURCES.md) - Comprehensive data sources guide
- [API Reference](api-reference/data-sources.md) - Full API documentation
- [Protocol Expansion Summary](PROTOCOL_EXPANSION_SUMMARY.md) - Technical details
- [Examples on GitHub](https://github.com/whanyu1212/QuantRL-Lab/tree/main/examples) - Working code examples

---

**Last Updated:** 2026-02-01
**Protocols Added:** SectorDataCapable, CompanyProfileCapable

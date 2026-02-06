# Documentation Index

Quick reference guide to all documentation related to the protocol-based architecture and recent protocol expansion.

---

## 📚 Core Architecture Documentation

### [ARCHITECTURE.md](ARCHITECTURE.md#design-decision-protocols-vs-abstract-base-classes-abc)
**Topic:** Design Decision: Protocols vs Abstract Base Classes (ABC)

**What you'll learn:**
- Why we chose protocols over traditional ABC inheritance
- Problems with ABC-only approach (diamond problem, rigid hierarchies)
- How the hybrid approach (DataSource ABC + Capability Protocols) works
- Real-world examples demonstrating extensibility benefits
- Comparison table of different approaches

**Key sections:**
- The Problem with ABC-Only Design
- The Protocol Solution
- Hybrid Approach: DataSource ABC + Capability Protocols
- Real-World Example: Adding New Capabilities
- When to Use Each Pattern

**Read this if:** You want to understand the fundamental architectural decision and rationale

---

### [ARCHITECTURAL_DECISIONS.md](ARCHITECTURAL_DECISIONS.md)
**Topic:** ADR-001: Protocols vs ABC for Data Source Capabilities

**What you'll learn:**
- Formal Architectural Decision Record (ADR)
- Problem statement and context
- Alternatives considered (with rejection reasons)
- Decision rationale and consequences
- Validation through testing

**Read this if:** You need a formal record of the architectural decision for team review or future reference

---

## 🔧 Protocol Implementation Documentation

### [DATA_SOURCES.md](DATA_SOURCES.md)
**Topic:** Complete guide to all data sources and protocols

**What you'll learn:**
- Overview of all data sources (Alpaca, YFinance, Alpha Vantage, FMP)
- Protocol-based architecture explanation
- Capability matrix (which source supports what)
- Protocol definitions and usage examples
- Best practices for each data source

**Key sections:**
- Protocol-Based Architecture
- Protocol Implementation (table showing which source implements which protocol)
- Protocol Definitions (code examples for each protocol)
- Data Source Details (comprehensive guide for each provider)

**Read this if:** You need comprehensive information about data sources and their capabilities

---

### [NEW_PROTOCOLS_REFERENCE.md](NEW_PROTOCOLS_REFERENCE.md)
**Topic:** Quick reference for SectorDataCapable and CompanyProfileCapable

**What you'll learn:**
- Protocol interface definitions
- Usage examples
- Common sectors and industries (for FMP)
- Company profile available fields
- Feature detection patterns
- Use cases (sector rotation, stock screening, trend analysis)

**Read this if:** You need a quick reference for the newly added protocols

---

### [API Reference - Data Sources](api-reference/data-sources.md)
**Topic:** Auto-generated API documentation

**What you'll learn:**
- Complete API documentation for all data sources
- Method signatures and parameters
- Return types and expected formats

**Read this if:** You need detailed API reference while coding

---

## 📝 Implementation Summaries

### [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
**Topic:** Complete summary of protocol expansion implementation

**What you'll learn:**
- What was implemented (new protocols, code changes, tests)
- Protocol matrix showing all capabilities
- Files modified/created
- Test coverage and validation results
- Benefits achieved
- Usage examples

**Read this if:** You want a complete technical summary of what was done

---

### [PROTOCOL_EXPANSION_SUMMARY.md](PROTOCOL_EXPANSION_SUMMARY.md)
**Topic:** Technical details of SectorDataCapable and CompanyProfileCapable addition

**What you'll learn:**
- Protocol definitions
- Implementation changes in FMP loader
- Test updates
- Documentation updates
- Protocol matrix
- Usage examples

**Read this if:** You need technical details about the specific protocol expansion

---

## 💻 Code Examples

### [https://github.com/whanyu1212/QuantRL-Lab/blob/main/examples/protocol_demonstration.py](https://github.com/whanyu1212/QuantRL-Lab/blob/main/examples/protocol_demonstration.py)
**Topic:** Runtime protocol detection demonstration

**What you'll learn:**
- How to check protocol support with `isinstance()`
- How to use `supports_feature()` method
- Conditional usage based on capabilities
- Feature discovery patterns

**Read this if:** You want working code examples of protocol-based feature detection

---

### [https://github.com/whanyu1212/QuantRL-Lab/blob/main/examples/fetch_fmp_data.py](https://github.com/whanyu1212/QuantRL-Lab/blob/main/examples/fetch_fmp_data.py)
**Topic:** Complete FMP data source usage examples

**What you'll learn:**
- How to fetch historical OHLCV data (daily and intraday)
- How to fetch analyst grades and ratings
- How to fetch company profiles
- How to fetch sector/industry performance data

**Read this if:** You need working examples of FMP API usage

---

## 🧪 Test Files

### [https://github.com/whanyu1212/QuantRL-Lab/blob/main/tests/data/test_protocols.py](https://github.com/whanyu1212/QuantRL-Lab/blob/main/tests/data/test_protocols.py)
**Topic:** Protocol conformance tests

**What you'll learn:**
- How to test protocol implementation
- How to verify method signatures
- How to test feature detection
- Positive and negative test patterns

**Read this if:** You're writing tests for new protocols or data sources

---

## 📊 Quick Reference Tables

### Protocol Implementation Matrix

| Data Source | Historical | Live | News | Fundamental | Macro | Analyst | Sector | Company Profile | Streaming |
|-------------|-----------|------|------|-------------|-------|---------|--------|-----------------|-----------|
| **FMP** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **YFinance** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Alpaca** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Alpha Vantage** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

### All Available Protocols

| Protocol | Purpose | Key Methods |
|----------|---------|-------------|
| **HistoricalDataCapable** | Historical OHLCV data | `get_historical_ohlcv_data()` |
| **LiveDataCapable** | Real-time quotes/trades | `get_latest_quote()`, `get_latest_trade()` |
| **NewsDataCapable** | News articles | `get_news_data()` |
| **FundamentalDataCapable** | Financial statements | `get_fundamental_data()` |
| **MacroDataCapable** | Economic indicators | `get_macro_data()` |
| **AnalystDataCapable** | Analyst ratings | `get_historical_grades()`, `get_historical_rating()` |
| **SectorDataCapable** | Sector/industry performance | `get_historical_sector_performance()`, `get_historical_industry_performance()` |
| **CompanyProfileCapable** | Company metadata | `get_company_profile()` |
| **StreamingCapable** | WebSocket streaming | `subscribe_to_updates()`, `start_streaming()`, `stop_streaming()` |
| **ConnectionManaged** | Connection lifecycle | `connect()`, `disconnect()`, `is_connected()` |

---

## 🎯 Documentation by Use Case

### "I want to understand why we use protocols instead of inheritance"
1. Read [ARCHITECTURE.md - Design Decision](ARCHITECTURE.md#design-decision-protocols-vs-abstract-base-classes-abc)
2. Review [ARCHITECTURAL_DECISIONS.md](ARCHITECTURAL_DECISIONS.md)
3. Look at code examples in [protocol_demonstration.py](https://github.com/whanyu1212/QuantRL-Lab/blob/main/examples/protocol_demonstration.py)

### "I want to use FMP's sector/industry data"
1. Quick start: [NEW_PROTOCOLS_REFERENCE.md](NEW_PROTOCOLS_REFERENCE.md)
2. Complete guide: [DATA_SOURCES.md - FMP Section](DATA_SOURCES.md#financial-modeling-prep-fmp)
3. Working example: [fetch_fmp_data.py](https://github.com/whanyu1212/QuantRL-Lab/blob/main/examples/fetch_fmp_data.py)

### "I want to add a new data source"
1. Read [ARCHITECTURE.md - Protocol Pattern](ARCHITECTURE.md#protocol-pattern-in-action)
2. Review [DATA_SOURCES.md - Protocol-Based Architecture](DATA_SOURCES.md#protocol-based-architecture)
3. Look at existing implementations in `src/quantrl_lab/data/sources/`
4. Study tests in [test_protocols.py](https://github.com/whanyu1212/QuantRL-Lab/blob/main/tests/data/test_protocols.py)

### "I want to add a new capability/protocol"
1. Read [IMPLEMENTATION_SUMMARY.md - How Protocols Enable Extensibility](IMPLEMENTATION_SUMMARY.md#3-how-protocols-enable-extensibility)
2. Review [PROTOCOL_EXPANSION_SUMMARY.md](PROTOCOL_EXPANSION_SUMMARY.md)
3. Follow the pattern in `src/quantrl_lab/data/interface.py`
4. Add tests following pattern in `https://github.com/whanyu1212/QuantRL-Lab/blob/main/tests/data/test_protocols.py`

### "I want to check what capabilities a data source has"
```python
from quantrl_lab.data.sources import FMPDataSource
from quantrl_lab.data.interface import SectorDataCapable

loader = FMPDataSource()

# Method 1: isinstance check
if isinstance(loader, SectorDataCapable):
    print("Supports sector data!")

# Method 2: supports_feature method
if loader.supports_feature("sector_data"):
    print("Supports sector data!")

# Method 3: Get all features
features = loader.supported_features
print(f"All features: {features}")
```

---

## 🔗 External References

### Python Documentation
- [PEP 544 - Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [typing.Protocol documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Abstract Base Classes](https://docs.python.org/3/library/abc.html)

### Related Patterns
- [Architectural Decision Records (ADR) pattern](https://adr.github.io/)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
- [Protocol-based design in Python](https://peps.python.org/pep-0544/#rationale-and-goals)

---

## 📁 File Organization

```
QuantRL-Lab/
├── docs/
│   ├── ARCHITECTURE.md              # Main architecture doc (includes Protocol vs ABC)
│   ├── DATA_SOURCES.md              # Complete data sources guide
│   ├── NEW_PROTOCOLS_REFERENCE.md   # Quick reference for new protocols
│   └── api-reference/
│       └── data-sources.md          # API reference
│
├── src/quantrl_lab/data/
│   ├── interface.py                 # Protocol definitions
│   └── sources/
│       ├── fmp_loader.py           # FMP implementation
│       ├── alpaca_loader.py        # Alpaca implementation
│       ├── yfinance_loader.py      # YFinance implementation
│       └── alpha_vantage_loader.py # Alpha Vantage implementation
│
├── tests/data/
│   ├── test_protocols.py           # Protocol conformance tests
│   └── test_data_sources.py        # Data source unit tests
│
├── examples/
│   ├── protocol_demonstration.py   # Protocol detection examples
│   └── fetch_fmp_data.py          # FMP usage examples
│
└── Root documentation/
    ├── ARCHITECTURAL_DECISIONS.md   # Formal ADR
    ├── IMPLEMENTATION_SUMMARY.md    # Complete implementation summary
    ├── PROTOCOL_EXPANSION_SUMMARY.md # Protocol expansion details
    └── DOCUMENTATION_INDEX.md       # This file
```

---

## ✅ Verification Checklist

Use this checklist to verify protocol implementation:

- [ ] Protocol defined in `src/quantrl_lab/data/interface.py`
- [ ] Protocol decorated with `@runtime_checkable`
- [ ] Data source class lists protocol in inheritance
- [ ] Required methods implemented in data source
- [ ] Tests added to `https://github.com/whanyu1212/QuantRL-Lab/blob/main/tests/data/test_protocols.py`
- [ ] Protocol added to `DataSource.supported_features` property
- [ ] Documentation updated in `DATA_SOURCES.md`
- [ ] Examples added or updated
- [ ] All tests pass (`uv run pytest https://github.com/whanyu1212/QuantRL-Lab/blob/main/tests/data/test_protocols.py`)
- [ ] MkDocs builds successfully (`uv run mkdocs build`)

---

**Last Updated:** 2026-02-01
**Maintainer:** QuantRL-Lab Team

---

**Quick Links:**
- [Main README](https://github.com/whanyu1212/QuantRL-Lab/blob/main/README.md)
- [Contributing Guidelines](https://github.com/whanyu1212/QuantRL-Lab/blob/main/CONTRIBUTING.md)
- [CLAUDE.md - Developer Guide](https://github.com/whanyu1212/QuantRL-Lab/blob/main/CLAUDE.md)
- [AGENTS.md - Architecture Guide](https://github.com/whanyu1212/QuantRL-Lab/blob/main/AGENTS.md)

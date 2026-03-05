# Architectural Decision Records (ADR)

This document records key architectural decisions made in QuantRL-Lab with rationale and consequences.

---

## ADR-001: Protocols vs Abstract Base Classes for Data Source Capabilities

**Date:** 2026-02-01
**Status:** Accepted
**Context:** Design of data source capability system

### Problem Statement

QuantRL-Lab supports multiple data providers (Alpaca, YFinance, Alpha Vantage, FMP) with different capabilities:
- Some provide historical data only (YFinance)
- Some provide live data and streaming (Alpaca)
- Some provide fundamental/macro data (Alpha Vantage)
- Some provide analyst data and sector performance (FMP)

**Challenge:** How do we design a type-safe, extensible system that allows:
1. Data sources to implement any combination of capabilities
2. Runtime feature detection
3. Type checker validation
4. Easy addition of new capabilities without modifying existing code

### Decision

Use a **hybrid approach** combining Abstract Base Classes with Protocol-based capability detection:

```python
# Base ABC for common infrastructure
class DataSource(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str: ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    @property
    def supported_features(self) -> List[str]:
        # Auto-detects which protocols are implemented
        ...

# Capability protocols for optional features
@runtime_checkable
class HistoricalDataCapable(Protocol):
    def get_historical_ohlcv_data(self, ...): ...

@runtime_checkable
class LiveDataCapable(Protocol):
    def get_latest_quote(self, ...): ...

@runtime_checkable
class AnalystDataCapable(Protocol):
    def get_historical_grades(self, ...): ...
    def get_historical_rating(self, ...): ...

# Implementation
class FMPDataSource(
    DataSource,  # Inherits infrastructure
    HistoricalDataCapable,  # Structural typing
    AnalystDataCapable,  # Structural typing
):
    # Just implement the methods - protocols satisfied automatically
    pass
```

### Alternatives Considered

#### Alternative 1: ABC-Only Approach
```python
class HistoricalDataSource(ABC):
    @abstractmethod
    def get_historical_ohlcv_data(self, ...): pass

class AlpacaDataLoader(DataSource, HistoricalDataSource, LiveDataSource, NewsDataSource):
    # Multiple inheritance complexity
    pass
```

**Rejected because:**
- ❌ Multiple inheritance leads to diamond problem
- ❌ Rigid inheritance hierarchy
- ❌ Cannot adapt third-party libraries without modification
- ❌ Adding new capabilities requires modifying class hierarchy

#### Alternative 2: Interface Classes (Java-style)
```python
class IHistoricalData:
    def get_historical_ohlcv_data(self, ...):
        raise NotImplementedError

class AlpacaDataLoader(IHistoricalData, ILiveData, INewsData):
    # All methods must be implemented
    pass
```

**Rejected because:**
- ❌ No type safety (isinstance checks don't work properly)
- ❌ Still requires multiple inheritance
- ❌ Verbose boilerplate for each interface

#### Alternative 3: Pure Duck Typing
```python
# Just implement methods, no protocols
class AlpacaDataLoader:
    def get_historical_ohlcv_data(self, ...): ...

# Check with hasattr
if hasattr(loader, 'get_historical_ohlcv_data'):
    data = loader.get_historical_ohlcv_data(...)
```

**Rejected because:**
- ❌ No type safety
- ❌ IDE autocomplete doesn't work
- ❌ Fragile (typos in method names)
- ❌ No documentation of intended interface

### Rationale for Chosen Approach

**Why Hybrid (DataSource ABC + Capability Protocols)?**

1. **DataSource ABC provides:**
   - Common infrastructure (connect/disconnect, source_name)
   - Shared utility methods (supported_features)
   - Single inheritance point for all data sources

2. **Capability Protocols provide:**
   - Flexible composition of optional features
   - Structural subtyping (duck typing with type safety)
   - No multiple inheritance issues
   - Easy to add new capabilities

**Real-world example of extensibility:**

When FMP added sector/industry performance data:

```python
# Step 1: Define new protocol (no changes to existing code)
@runtime_checkable
class SectorDataCapable(Protocol):
    def get_historical_sector_performance(self, ...): ...
    def get_historical_industry_performance(self, ...): ...

# Step 2: Add methods to FMP (automatically satisfies protocol)
class FMPDataSource(DataSource):
    def get_historical_sector_performance(self, sector):
        # Implementation
        pass

# Step 3: Works immediately - no inheritance changes!
if isinstance(fmp, SectorDataCapable):  # ✅ True
    data = fmp.get_historical_sector_performance("Technology")
```

### Consequences

**Positive:**
- ✅ Type-safe feature detection via `isinstance(obj, Protocol)`
- ✅ No diamond problem or method resolution order issues
- ✅ Easy to add new capabilities without modifying existing code
- ✅ IDE autocomplete and type checker support
- ✅ Clean single inheritance hierarchy
- ✅ Can adapt third-party libraries through structural typing
- ✅ Auto-discovery via `supported_features` property

**Negative:**
- ⚠️ Requires understanding of both ABC and Protocol patterns
- ⚠️ Developers must remember to use `@runtime_checkable` decorator
- ⚠️ Python 3.8+ required (typing.Protocol added in 3.8)

**Neutral:**
- Protocol compliance is checked structurally (method signatures must match)
- Protocols are checked at runtime with `isinstance()` or statically with type checkers

### Validation

**Test coverage:**
- 52 protocol conformance tests in `tests/data/test_protocols.py`
- Runtime protocol checking verified for all data sources
- Feature detection via `supported_features` tested
- Method signature validation for all protocols

**Real usage:**
```python
# Example 1: Conditional feature usage
from quantrl_lab.data.sources import FMPDataSource
from quantrl_lab.data.interface import SectorDataCapable

loader = FMPDataSource()

if isinstance(loader, SectorDataCapable):
    # Type checker knows this is safe
    sector_data = loader.get_historical_sector_performance("Energy")

# Example 2: Feature discovery
features = loader.supported_features
# ['historical_bars', 'analyst_data', 'sector_data', 'company_profile']

if loader.supports_feature("sector_data"):
    # Use sector data features
    pass
```

### References

- [PEP 544 - Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [Python typing documentation - Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Effective Python Item 43: Consider Protocols and Duck Typing](https://effectivepython.com/)
- [ARCHITECTURE.md - Design Decision: Protocols vs ABC](ARCHITECTURE.md#design-decision-protocols-vs-abstract-base-classes-abc)

### Related Decisions

- **ADR-002 (Potential):** When to add new capability protocols vs extending existing ones
- **ADR-003 (Potential):** Protocol versioning strategy for backward compatibility

---

## Future ADRs

### Candidates for Documentation:

1. **Strategy Pattern for Environment Components**
   - Why inject action/reward/observation strategies vs inheritance?
   - Benefits: testability, composability, separation of concerns

2. **Registry Pattern for Technical Indicators**
   - Why decorator-based registration vs manual registration?
   - Benefits: auto-discovery, plugin architecture, extensibility

3. **Utility Module Refactoring**
   - Why centralized utilities vs duplicated code in each loader?
   - Benefits: DRY principle, consistency, maintainability

4. **Gymnasium-based Environment Design**
   - Why Gymnasium over raw OpenAI Gym?
   - Benefits: active maintenance, type hints, modern API

---

**Last Updated:** 2026-02-01
**Maintainer:** QuantRL-Lab Team

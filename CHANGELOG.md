# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CLAUDE.md**: Guidance file for Claude Code (claude.ai/code) with quick reference commands and architecture overview
- **AGENTS.md**: Comprehensive guide for AI assistants working with the codebase
  - Architecture patterns (protocol-based design, registry pattern, strategy injection)
  - Common workflows and development gotchas
  - Essential commands for uv, testing, and notebooks
- **Comprehensive test suite** for data module:
  - `tests/data/test_indicators.py` - 40 tests for IndicatorRegistry and all technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Stochastic, OBV)
  - `tests/data/test_data_sources.py` - 34 unit tests for data loaders with mocked APIs
  - `tests/data/test_data_sources_integration.py` - 16 integration tests using real APIs (YFinance, Alpaca, Alpha Vantage)
- **Integration test marker** (`@pytest.mark.integration`) to separate API-dependent tests from unit tests
- **Protocol-based data interface improvements**:
  - Standardized return type to `pd.DataFrame` for `NewsDataCapable`, `FundamentalDataCapable`, and `MacroDataCapable` protocols
  - Runtime capability checks with `isinstance()` for better error handling
  - Added defensive checks in `DataSourceRegistry` to validate protocol implementation before method calls
- **Enhanced type hints**: Added proper type annotations to `DataSourceRegistry.__init__()` for better IDE support

### Changed
- **Updated CI/CD workflows** (`.github/workflows/ci.yaml`, `.github/workflows/cd.yaml`):
  - Migrated from Poetry to uv for consistency with local development
  - Added `-m "not integration"` to exclude API-dependent tests
  - Updated import paths from `custom_envs` to `environments`
  - Upgraded GitHub Actions to v4 (checkout, codecov)
  - Use `uv build` and `uv publish` for package building/publishing
  - Removed manual Poetry caching (uv handles this automatically)
- **Migrated from Poetry to uv** for faster dependency management
  - 10-100x faster package resolution and installation
  - Maintains compatibility with existing `pyproject.toml`
  - Updated all documentation and CI/CD workflows
- **Restructured data module**:
  - Renamed `loaders/` → `sources/` for clarity
  - Separated data processing logic into dedicated `processors/` submodule
  - Enhanced data source interface with `DataSource` abstract base class
  - Improved feature detection via `supported_features` property
- **Reorganized experiment workflows**:
  - Grouped offline experiments under `experiments/` (backtesting, feature engineering, tuning, screening)
  - Separated production code under `deployment/` (trading only)
  - Moved `screening/` from `deployment/` → `experiments/` (hedge pair discovery is research, not production)
  - Moved `feature_selection/` → `feature_engineering/` for accuracy
- **Refined documentation**: Updated README with system architecture diagrams (Mermaid)
  - Protocol pattern illustration
  - Registry pattern workflow
  - Strategy injection design

### Fixed
- Protocol return type consistency for better downstream processing
- Missing error handling when data sources don't implement required protocols
- **YFinance loader bug**: Fixed `period=timeframe` → `interval=timeframe` parameter in `get_historical_ohlcv_data()` which was causing API errors
- **Alpha Vantage free tier compatibility**: Fixed API failures due to premium-only features
  - Changed default `outputsize` from `full` to `compact` (last 100 data points)
  - Added automatic rate limiting (1.2s between requests) to respect 1 req/sec burst limit
  - Added `Information` key logging to surface API error messages (rate limits, premium requirements)
  - Updated docstrings and example file with accurate free tier limitations

### Security

### Refactor
- **Test directory restructure**: Reorganized `tests/` to mirror `src/quantrl_lab/` structure
  - `tests/data/` - Data module tests (indicators, sources)
  - `tests/environments/stock/` - Stock environment tests (env, portfolio, action, reward)
- **Data Module Cleanup**: Simplified filenames in `src/quantrl_lab/data/` to remove redundancy.
  - Renamed `indicators/indicator_registry.py` -> `indicators/registry.py`
  - Renamed `indicators/technical_indicators.py` -> `indicators/technical.py`
  - Renamed `sources/alpaca_loader.py` -> `sources/alpaca.py`
  - Renamed `sources/yfinance_loader.py` -> `sources/yfinance.py`
  - Renamed `sources/alpha_vantage_loader.py` -> `sources/alpha_vantage.py`
  - Renamed `data_source_registry.py` -> `source_registry.py`

## [0.1.0] - 2025-10-16

### Added
- Initial release of QuantRL-Lab
- Gymnasium-compatible trading environments for stocks, crypto, and forex
- Modular strategy system with customizable actions, rewards, and observations
- Integration with Stable-Baselines3 and other RL frameworks
- Comprehensive backtesting framework with performance metrics
- Technical indicators library for feature engineering
- Support for multiple data sources (yfinance, Alpaca, Alpha Vantage)
- Optuna integration for hyperparameter tuning
- LLM-based hedge screening capabilities
- Complete CI/CD pipeline for automated testing and PyPI publishing
- Extensive documentation and examples

### Documentation
- Setup guide for development environment
- Publishing workflow documentation
- Example notebooks for backtesting, feature selection, and tuning
- API documentation

[Unreleased]: https://github.com/whanyu1212/QuantRL-Lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/whanyu1212/QuantRL-Lab/releases/tag/v0.1.0

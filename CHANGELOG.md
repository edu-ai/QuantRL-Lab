# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **AGENTS.md**: Comprehensive guide for AI assistants working with the codebase
  - Architecture patterns (protocol-based design, registry pattern, strategy injection)
  - Common workflows and development gotchas
  - Essential commands for uv, testing, and notebooks
- **Protocol-based data interface improvements**:
  - Standardized return type to `pd.DataFrame` for `NewsDataCapable`, `FundamentalDataCapable`, and `MacroDataCapable` protocols
  - Runtime capability checks with `isinstance()` for better error handling
  - Added defensive checks in `DataSourceRegistry` to validate protocol implementation before method calls
- **Enhanced type hints**: Added proper type annotations to `DataSourceRegistry.__init__()` for better IDE support

### Changed
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
  - Grouped offline experiments under `experiments/` (backtesting, feature engineering, tuning)
  - Separated production code under `deployment/` (trading, screening)
  - Moved `feature_selection/` → `feature_engineering/` for accuracy
- **Refined documentation**: Updated README with system architecture diagrams (Mermaid)
  - Protocol pattern illustration
  - Registry pattern workflow
  - Strategy injection design

### Fixed
- Protocol return type consistency for better downstream processing
- Missing error handling when data sources don't implement required protocols

### Security

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

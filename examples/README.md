# QuantRL-Lab Examples

This folder contains example scripts demonstrating key features of QuantRL-Lab.

## Directory Structure

- `data_sources/`: Scripts demonstrating how to fetch data from different providers.
- `pipelines/`: End-to-end data processing pipelines using the `DataProcessor`.
- `features/`: Examples focusing on specific features like technical indicators.
- `core_concepts/`: Demonstrations of core architectural concepts like protocols.

## Categories

### Data Pipelines (`examples/pipelines/`)
| Script | Description | API Key Required |
|--------|-------------|------------------|
| `data_processing_basic.py` | **Start here!** Basic pipeline with indicators | No |
| `data_processing_advanced.py` | Multiple indicators, custom params, train/test split | No |
| `data_processing_param_grid.py` | Parameter grid search and 3-way splits | No |
| `data_processing_date_split.py` | Date-based splitting and walk-forward validation | No |
| `data_processing_with_sentiment.py` | Sentiment analysis integration | Yes (Alpaca) |
| `data_processing_complete.py` | **All features showcase** - comprehensive example | Yes (Alpaca) |
| `data_processing_file_config.py` | Loading configurations from YAML/JSON files | No |

### Data Sources (`examples/data_sources/`)
| Script | Data Source | API Key Required |
|--------|-------------|------------------|
| `fetch_yfinance_data.py` | Yahoo Finance | No |
| `fetch_alpaca_data.py` | Alpaca | Yes |
| `fetch_alphavantage_data.py` | Alpha Vantage | Yes |
| `fetch_fmp_data.py` | Financial Modeling Prep | Yes |

### Feature Examples (`examples/features/`)
| Script | Description | API Key Required |
|--------|-------------|------------------|
| `indicators_usage.py` | Technical indicator registry usage | No |
| `indicator_selection_workflow.py` | Alpha-driven indicator selection via `AlphaSelector` | No |

### Alpha Research (`examples/alpha_research/`)
| Script | Description | API Key Required |
|--------|-------------|------------------|
| `run_alpha_workflow.py` | Full alpha pipeline: selection, ensemble, robustness, HTML report | No |

### Core Concepts (`examples/core_concepts/`)
| Script | Description | API Key Required |
|--------|-------------|------------------|
| `protocol_demonstration.py` | Data source protocol capabilities | No |

### End-to-End Training (`examples/end_to_end/`)
| Script | Description | API Key Required |
|--------|-------------|------------------|
| `single_asset/train_single_symbol.py` | Full pipeline: data → alpha → training → evaluation | No |
| `single_asset/train_multi_symbol.py` | Multi-symbol vectorized training with async data fetch | Optional |
| `single_asset/tune_single_symbol.py` | Hyperparameter tuning via Optuna (3-way split) | No |

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

### Data Processing Pipelines

**Recommended order for learning:**

```bash
# 1. Start here - basic concepts
uv run python examples/pipelines/data_processing_basic.py

# 2. Learn advanced features
uv run python examples/pipelines/data_processing_advanced.py

# 3. Explore parameter variations
uv run python examples/pipelines/data_processing_param_grid.py

# 4. Master date-based splitting
uv run python examples/pipelines/data_processing_date_split.py

# 5. Add sentiment analysis (requires Alpaca API key)
uv run python examples/pipelines/data_processing_with_sentiment.py

# 6. See everything together (requires Alpaca API key)
uv run python examples/pipelines/data_processing_complete.py
```

### Data Sources

```bash
# Yahoo Finance (no API key needed)
uv run python examples/data_sources/fetch_yfinance_data.py

# Alpaca (requires API key)
uv run python examples/data_sources/fetch_alpaca_data.py

# Alpha Vantage (requires API key)
uv run python examples/data_sources/fetch_alphavantage_data.py

# Financial Modeling Prep (requires API key)
uv run python examples/data_sources/fetch_fmp_data.py
```

### Feature & Alpha Research

```bash
# Technical indicator registry
uv run python examples/features/indicators_usage.py

# Alpha-driven indicator selection
uv run python examples/features/indicator_selection_workflow.py

# Full alpha research pipeline (selection, ensemble, robustness, HTML report)
uv run python examples/alpha_research/run_alpha_workflow.py

# Protocol capabilities
uv run python examples/core_concepts/protocol_demonstration.py
```

### End-to-End Training

```bash
# Single stock: data → alpha research → training → evaluation
uv run python examples/end_to_end/single_asset/train_single_symbol.py

# Multi-symbol: vectorized training across multiple stocks
uv run python examples/end_to_end/single_asset/train_multi_symbol.py

# Hyperparameter tuning with Optuna
uv run python examples/end_to_end/single_asset/tune_single_symbol.py
```

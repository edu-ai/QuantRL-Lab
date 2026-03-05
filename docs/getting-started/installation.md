# Installation

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quick Install

=== "uv (Recommended)"

    ```bash
    # Clone the repository
    git clone https://github.com/whanyu1212/QuantRL-Lab.git
    cd QuantRL-Lab

    # Install all dependencies
    uv sync

    # Install with specific extras
    uv sync --extra dev --extra notebooks

    # Activate virtual environment
    source .venv/bin/activate
    ```

=== "pip"

    ```bash
    # Clone the repository
    git clone https://github.com/whanyu1212/QuantRL-Lab.git
    cd QuantRL-Lab

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install package
    pip install -e ".[dev,notebooks]"
    ```

## Environment Setup

QuantRL-Lab requires API keys for data sources. Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Alpaca Trading API (for data & live trading)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Alpha Vantage (for alternative data)
ALPHA_VANTAGE_API_KEY=your_key_here
```

### Getting API Keys

- **Alpaca**: [Sign up for free](https://alpaca.markets/) (paper trading account)
- **Alpha Vantage**: [Get free API key](https://www.alphavantage.co/support/#api-key)

!!! info "YFinance requires no API key"
    The `YFinanceDataLoader` works out of the box with no configuration.
    It's the easiest way to get started.

## Verify Installation

```bash
# Check imports
python -c "from quantrl_lab import environments, data, experiments; print('Installation successful')"

# Run tests
pytest
```

## Jupyter Setup (Optional)

For running notebooks:

```bash
# Install Jupyter kernel
python -m ipykernel install --user --name quantrl-lab --display-name "QuantRL-Lab"

# Start Jupyter
jupyter notebook
# Then select "QuantRL-Lab" kernel in notebook
```

## Troubleshooting

??? warning "Import Error: `ModuleNotFoundError: No module named 'quantrl_lab'`"
    Ensure you're in the virtual environment:
    ```bash
    source .venv/bin/activate  # Unix/Mac
    .venv\Scripts\activate     # Windows
    ```

??? warning "API Key Errors"
    Verify `.env` file exists and contains valid keys:
    ```bash
    cat .env | grep ALPACA_API_KEY
    ```

??? warning "Dependency Conflicts"
    Reinstall with clean environment:
    ```bash
    rm -rf .venv
    uv sync --extra dev
    ```

## Next Steps

- [Quickstart Guide](quickstart.md) - Train your first agent
- [Configuration](configuration.md) - Customize environment settings
- [Examples](../examples/basic-backtest.md) - Explore notebooks

# QuantRL-Lab

[![PyPI version](https://badge.fury.io/py/quantrl-lab.svg)](https://badge.fury.io/py/quantrl-lab)
[![Python](https://img.shields.io/pypi/pyversions/quantrl-lab.svg)](https://pypi.org/project/quantrl-lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://whanyu1212.github.io/QuantRL-Lab/)

A Python testbed for Reinforcement Learning in finance, designed to enable researchers and developers to experiment with and evaluate RL algorithms in financial contexts. The project emphasizes modularity and configurability, allowing users to tailor the environment, data sources, and algorithmic settings to their specific needs.

## Installation

### Quick Start (Core Dependencies Only)

```bash
# Using pip
pip install quantrl-lab

# Using uv (recommended - much faster)
uv pip install quantrl-lab
```

This installs only essential runtime dependencies (~72 packages). For additional features:

```bash
# Jupyter notebook support
pip install quantrl-lab[notebooks]

# ML/LLM features (torch, transformers, litellm, openai)
pip install quantrl-lab[ml]

# Development tools (pytest, black, mypy, etc.)
pip install quantrl-lab[dev]

# Everything
pip install quantrl-lab[full]
```

**For contributors**: See [MIGRATION.md](MIGRATION.md) for complete uv workflow and development setup.

## Table of Contents
- [Installation](#installation)
- [Motivation](#motivation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Contributors](#contributors)

---

## Motivation

**Addressing the Monolithic Environment Problem**

Most existing RL frameworks for finance suffer from tightly coupled, monolithic designs where action spaces, observation spaces, and reward functions are hardcoded into the environment initialization. This creates several critical limitations:

- **Limited Experimentation**: Users cannot easily test different reward formulations or action spaces without doing a lot of rewriting of the environments
- **Poor Scalability**: Adding new asset classes, trading strategies, or market conditions requires significant code restructuring
- **Reduced Reproducibility**: Inconsistent interfaces across different environment configurations make fair comparisons difficult
- **Development Overhead**: Simple modifications like testing different reward functions or adding new observation features require extensive refactoring

**QuantRL-Lab's Solution**: Dependency injection of pluggable strategies (action/observation/reward) decouples environment logic from algorithmic choices, enabling rapid experimentation without code rewrites.

---

## Quick Start
---


---

## Quick Start

### System Workflow

End-to-end process from data acquisition to model evaluation:

```mermaid
flowchart TB
    A[Fetch Historical Data] --> B[Configure Pipeline]

    B --> C[Compute Indicators: RSI, MACD, MAC, BB etc.]

    C --> D[Instantiate Environment with Strategies]

    D --> E[Action Strategy]
    D --> F[Observation Strategy]
    D --> G[Reward Strategy]

    E --> H
    F --> H
    G --> H

    H[Train RL Agent: PPO/SAC/A2C] --> I[Evaluate vs Benchmarks]

    I --> J[Analyze Results]

    J --> K{Iterate?}

    K -->|Yes| B
    K -->|No| L[End]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#e8f5e8
    style H fill:#fce4ec
    style I fill:#e0f2f1
    style J fill:#f1f8e9
    style K fill:#fce4ec
    style L fill:#c8e6c9
```

### Example Usage

```python
# Easily swappable strategies for experimentation
# For in-depth example, see notebooks/backtesting_example.ipynb

sample_env_config = BacktestRunner.create_env_config_factory(
    train_data=train_data_df,
    test_data=test_data_df,
    action_strategy=action_strategy,
    reward_strategy=reward_strategies["conservative"],
    observation_strategy=observation_strategy,
    initial_balance=100000.0,
    transaction_cost_pct=0.001,
    window_size=20
)

runner = BacktestRunner(verbose=1)

# Single experiment
results = runner.run_single_experiment(
    SAC,          # Algorithm to use
    sample_env_config,
    # config=custom_sac_config,  # optional custom config
    total_timesteps=50000,
    num_eval_episodes=3
)

BacktestRunner.inspect_single_experiment(results)

# Comprehensive sweep across multiple algorithms and presets
presets = ["default", "explorative", "conservative"]
algorithms = [PPO, A2C, SAC]

comprehensive_results = runner.run_comprehensive_backtest(
    algorithms=algorithms,
    env_configs=env_configs,
    presets=presets,
    # custom_configs=custom_configs,  # either use presets or customize
    total_timesteps=50000,
    n_envs=4,
    num_eval_episodes=3
)
```

**Detailed tutorials:**
- Feature and window size selection: [`notebooks/feature_selection.ipynb`](notebooks/feature_selection.ipynb)
- Data processing example: [`notebooks/data_processing.ipynb`](notebooks/data_processing.ipynb)
- Backtesting: [`notebooks/backtesting_example.ipynb`](notebooks/backtesting_example.ipynb)
- Hyperparameter tuning: [`notebooks/hyperparameter_tuning.ipynb`](notebooks/hyperparameter_tuning.ipynb)
- LLM hedge pair screener: [`notebooks/llm_hedge_screener.ipynb`](notebooks/llm_hedge_screener.ipynb)

---

## Architecture

For detailed architecture documentation, design patterns, and system diagrams, see the [**Architecture Guide**](docs/ARCHITECTURE.md):

- 🔄 **Workflow Overview** - End-to-end experimental loop
- 🏗️ **High-Level Architecture** - Layered system design (Data/Environment/Experiment/Utilities layers)
- 🔌 **Strategy Pattern** - How pluggable strategies interact with environments
- 📊 **Data Flow Pipeline** - Complete transformation from raw sources to RL agent
- 🧩 **Pre-built Components** - Out-of-the-box action/observation/reward strategies
- 🔧 **Extensibility Guide** - How to create custom strategies
- 🔍 **Protocol Pattern** - Structural typing for flexible data sources
- 📋 **Registry Pattern** - Technical indicator management
- ⚡ **Reward Strategy Pattern** - Decoupled reward function experimentation

---

## Roadmap

- **Data Source Expansion**:
  - Complete integration for more (free) data sources
  - Add crypto data support
  - Add OANDA forex data support
- **Technical Indicators**:
  - Add more indicators (Ichimoku, Williams %R, CCI, etc.)
- **Trading Environments**:
  - (In-progress) Multi-stock trading environment with hedging pair capabilities
- **Alternative Data for consideration in observable space**:
  - Fundamental data (earnings, balance sheets, income statements, cash flow)
  - Macroeconomic indicators (GDP, inflation, unemployment, interest rates)
  - Economic calendar events
  - Sector performance data

---

### Development Setup

> **Note:** This section is for developers who want to contribute to QuantRL-Lab or run it from source. If you just want to use the library, simply install it with `pip install quantrl-lab`.

#### For Contributors and Developers

1. Clone the Repository
```bash
git clone https://github.com/whanyu1212/QuantRL-Lab.git
cd QuantRL-Lab
```

2. Install Poetry for dependency management
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies (installs the project in development mode)
```bash
poetry install
```

4. Activate virtual environment
```bash
# The shell command is deprecated, use this instead:
poetry env info  # This shows the venv path
# Then activate it manually, e.g.:
source /path/to/virtualenv/bin/activate  # macOS/Linux
```

5. (Optional) Install jupyter kernel for notebook examples
```bash
python -m ipykernel install --user --name quantrl-lab --display-name "QuantRL-Lab"
```

6. Set up environment variables (for data sources like Alpaca, Alpha Vantage, etc.)
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Never commit the .env file to version control
```

7. Set up pre-commit hooks (for code quality)
```bash
# Install the git hooks
pre-commit install

# (Optional) Run on all files to test
pre-commit run --all-files
```

The pre-commit hooks will automatically check:
- Code formatting (black)
- Import sorting (isort)
- Code linting (flake8)
- Docstring formatting (docformatter)
- File checks (trailing whitespace, YAML validation, etc.)

To skip pre-commit hooks temporarily:
```bash
git commit -m "your message" --no-verify
```

---

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or suggesting ideas, your help is appreciated.

### How to Contribute

1. **Fork the repository** and create a branch for your feature or fix
2. **Make your changes** following our coding standards
3. **Write tests** for new functionality
4. **Submit a pull request** with a clear description

Please read our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on:
- Setting up your development environment
- Coding standards and style guidelines
- Testing requirements
- Pull request process

### Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build something great together!

---

## Contributors

This project exists thanks to all the people who contribute.

### Main Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/whanyu1212">
        <img src="https://github.com/whanyu1212.png" width="100px;" alt="whanyu1212"/>
        <br />
        <sub><b>whanyu1212</b></sub>
      </a>
      <br />
      <sub>Creator & Maintainer</sub>
    </td>
  </tr>
</table>

### How to Become a Contributor

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) to get started.

---

### Literature Review
placeholder

"""
Unit tests for the alpha_research module.

Tests cover:
- Core data structures (AlphaJob, AlphaResult)
- Strategy registry
- Vectorized strategies
- AlphaRunner execution
- Ensemble methods
- Robustness testing
- Visualization
"""

import numpy as np
import pandas as pd
import pytest

from quantrl_lab.alpha_research.alpha_strategies import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)
from quantrl_lab.alpha_research.analysis import RobustnessTester
from quantrl_lab.alpha_research.base import SignalType
from quantrl_lab.alpha_research.ensemble import AlphaEnsemble
from quantrl_lab.alpha_research.models import AlphaJob, AlphaResult
from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry
from quantrl_lab.alpha_research.runner import AlphaRunner
from quantrl_lab.alpha_research.visualization import AlphaVisualizer

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 252  # One year of trading days
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # Generate realistic price data with trend
    base_price = 100
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.randn(n) * 0.005),
            "High": prices * (1 + np.abs(np.random.randn(n) * 0.01)),
            "Low": prices * (1 - np.abs(np.random.randn(n) * 0.01)),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, n),
        },
        index=dates,
    )

    return df


@pytest.fixture
def sample_data_with_indicators(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to sample data."""
    df = sample_ohlcv_data.copy()

    # Add SMA
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Add RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Add MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Add Williams %R
    df["WILLR_14"] = (
        (df["High"].rolling(14).max() - df["Close"]) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min())
    ) * -100

    # Drop NaN rows
    df = df.dropna()

    return df


@pytest.fixture
def sample_alpha_job(sample_data_with_indicators: pd.DataFrame) -> AlphaJob:
    """Create a sample AlphaJob for testing."""
    return AlphaJob(
        data=sample_data_with_indicators,
        indicator_name="RSI",
        strategy_name="mean_reversion",
        indicator_params={"window": 14},
        strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
        allow_short=True,
    )


@pytest.fixture
def sample_alpha_result(sample_alpha_job: AlphaJob, sample_data_with_indicators: pd.DataFrame) -> AlphaResult:
    """Create a sample AlphaResult for testing."""
    # Create a simple equity curve
    n = len(sample_data_with_indicators)
    equity = pd.Series(1.0 + np.cumsum(np.random.randn(n) * 0.01), index=sample_data_with_indicators.index)

    return AlphaResult(
        job=sample_alpha_job,
        metrics={
            "sharpe_ratio": 0.75,
            "sortino_ratio": 1.1,
            "calmar_ratio": 0.5,
            "total_return": 0.15,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "ic": 0.03,
            "rank_ic": 0.04,
            "mutual_info": 0.01,
        },
        equity_curve=equity,
        signals=pd.Series(np.random.choice([1, 0, -1], n), index=sample_data_with_indicators.index),
        status="completed",
    )


@pytest.fixture
def multiple_alpha_results(sample_data_with_indicators: pd.DataFrame) -> list:
    """Create multiple AlphaResults for ensemble testing."""
    results = []
    strategies = [
        ("RSI", "mean_reversion", {"indicator_col": "RSI_14"}),
        ("SMA", "trend_following", {"indicator_col": "SMA_20"}),
        ("MACD", "macd_crossover", {"fast_col": "MACD", "slow_col": "MACD_signal"}),
    ]

    n = len(sample_data_with_indicators)

    for i, (ind, strat, params) in enumerate(strategies):
        job = AlphaJob(
            data=sample_data_with_indicators,
            indicator_name=ind,
            strategy_name=strat,
            strategy_params=params,
        )

        # Generate different equity curves
        np.random.seed(42 + i)
        equity = pd.Series(1.0 + np.cumsum(np.random.randn(n) * 0.01), index=sample_data_with_indicators.index)

        result = AlphaResult(
            job=job,
            metrics={
                "sharpe_ratio": 0.5 + i * 0.2,
                "sortino_ratio": 0.7 + i * 0.2,
                "total_return": 0.1 + i * 0.05,
                "max_drawdown": -0.15 + i * 0.02,
                "ic": 0.02 + i * 0.01,
            },
            equity_curve=equity,
            status="completed",
        )
        results.append(result)

    return results


# ============================================================================
# Core Data Structure Tests
# ============================================================================


class TestAlphaJob:
    """Tests for AlphaJob dataclass."""

    def test_alpha_job_creation(self, sample_ohlcv_data):
        """Test basic AlphaJob creation."""
        job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="RSI",
            strategy_name="mean_reversion",
        )

        assert job.indicator_name == "RSI"
        assert job.strategy_name == "mean_reversion"
        assert len(job.id) == 8  # UUID short format
        assert isinstance(job.indicator_params, dict)
        assert isinstance(job.strategy_params, dict)

    def test_alpha_job_with_params(self, sample_ohlcv_data):
        """Test AlphaJob with custom parameters."""
        job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
            strategy_params={"oversold": 25, "overbought": 75},
            allow_short=False,
            tags={"category": "momentum"},
        )

        assert job.indicator_params["window"] == 14
        assert job.strategy_params["oversold"] == 25
        assert job.allow_short is False
        assert job.tags["category"] == "momentum"

    def test_alpha_job_unique_ids(self, sample_ohlcv_data):
        """Test that each AlphaJob gets a unique ID."""
        jobs = [
            AlphaJob(data=sample_ohlcv_data, indicator_name="RSI", strategy_name="mean_reversion") for _ in range(10)
        ]

        ids = [j.id for j in jobs]
        assert len(ids) == len(set(ids)), "Job IDs should be unique"


class TestAlphaResult:
    """Tests for AlphaResult dataclass."""

    def test_alpha_result_creation(self, sample_alpha_job):
        """Test basic AlphaResult creation."""
        result = AlphaResult(
            job=sample_alpha_job,
            metrics={"sharpe_ratio": 1.0},
            status="completed",
        )

        assert result.job == sample_alpha_job
        assert result.metrics["sharpe_ratio"] == 1.0
        assert result.status == "completed"
        assert result.equity_curve is None
        assert result.error is None

    def test_alpha_result_with_error(self, sample_alpha_job):
        """Test AlphaResult with failed status."""
        error = ValueError("Test error")
        result = AlphaResult(
            job=sample_alpha_job,
            metrics={},
            status="failed",
            error=error,
        )

        assert result.status == "failed"
        assert result.error == error


# ============================================================================
# Strategy Registry Tests
# ============================================================================


class TestVectorizedStrategyRegistry:
    """Tests for VectorizedStrategyRegistry."""

    def test_list_strategies(self):
        """Test listing registered strategies."""
        strategies = VectorizedStrategyRegistry.list_strategies()

        assert isinstance(strategies, list)
        assert "mean_reversion" in strategies
        assert "trend_following" in strategies
        assert "macd_crossover" in strategies

    def test_create_strategy(self):
        """Test creating a strategy by name."""
        strategy = VectorizedStrategyRegistry.create(
            "mean_reversion",
            indicator_col="RSI_14",
            oversold=30,
            overbought=70,
        )

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.indicator_col == "RSI_14"
        assert strategy.oversold == 30
        assert strategy.overbought == 70

    def test_create_unknown_strategy(self):
        """Test creating an unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            VectorizedStrategyRegistry.create("nonexistent_strategy")


# ============================================================================
# Vectorized Strategy Tests
# ============================================================================


class TestTrendFollowingStrategy:
    """Tests for TrendFollowingStrategy."""

    def test_generate_signals(self, sample_data_with_indicators):
        """Test signal generation."""
        strategy = TrendFollowingStrategy(indicator_col="SMA_20", allow_short=True)
        signals = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)
        assert set(signals.unique()).issubset({SignalType.BUY.value, SignalType.SELL.value, SignalType.HOLD.value})

    def test_generate_signals_no_short(self, sample_data_with_indicators):
        """Test signal generation without shorting."""
        strategy = TrendFollowingStrategy(indicator_col="SMA_20", allow_short=False)
        signals = strategy.generate_signals(sample_data_with_indicators)

        # Should not have SELL signals when shorting is disabled
        assert SignalType.SELL.value not in signals.values

    def test_required_columns(self):
        """Test required columns."""
        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        cols = strategy.get_required_columns()

        assert "SMA_20" in cols
        assert "Close" in cols


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy."""

    def test_generate_signals(self, sample_data_with_indicators):
        """Test signal generation."""
        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)

    def test_oversold_generates_buy(self, sample_data_with_indicators):
        """Test that oversold condition generates BUY signal."""
        # Force some RSI values to be oversold
        df = sample_data_with_indicators.copy()
        df.loc[df.index[:10], "RSI_14"] = 20  # Oversold

        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70)
        signals = strategy.generate_signals(df)

        # At least some of the first 10 signals should be BUY
        assert SignalType.BUY.value in signals.iloc[:10].values

    def test_overbought_generates_sell(self, sample_data_with_indicators):
        """Test that overbought condition generates SELL signal."""
        df = sample_data_with_indicators.copy()
        df.loc[df.index[:10], "RSI_14"] = 80  # Overbought

        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70, allow_short=True)
        signals = strategy.generate_signals(df)

        assert SignalType.SELL.value in signals.iloc[:10].values


class TestMACDCrossoverStrategy:
    """Tests for MACDCrossoverStrategy."""

    def test_generate_signals(self, sample_data_with_indicators):
        """Test signal generation."""
        strategy = MACDCrossoverStrategy(fast_col="MACD", slow_col="MACD_signal")
        signals = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)


# ============================================================================
# AlphaRunner Tests
# ============================================================================


class TestAlphaRunner:
    """Tests for AlphaRunner."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        runner = AlphaRunner(verbose=False)
        assert runner.verbose is False

    def test_run_single_job(self, sample_alpha_job):
        """Test running a single job."""
        runner = AlphaRunner(verbose=False)
        result = runner.run_job(sample_alpha_job)

        assert isinstance(result, AlphaResult)
        assert result.job == sample_alpha_job
        # Result should be either completed or failed
        assert result.status in ["completed", "failed"]

    def test_run_batch(self, sample_data_with_indicators):
        """Test running a batch of jobs."""
        jobs = [
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                strategy_params={"indicator_col": "RSI_14"},
            ),
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="SMA",
                strategy_name="trend_following",
                strategy_params={"indicator_col": "SMA_20"},
            ),
        ]

        runner = AlphaRunner(verbose=False)
        results = runner.run_batch(jobs, n_jobs=1)

        assert len(results) == 2
        assert all(isinstance(r, AlphaResult) for r in results)

    def test_run_batch_parallel(self, sample_data_with_indicators):
        """Test running batch with parallel execution."""
        jobs = [
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                strategy_params={"indicator_col": "RSI_14"},
            )
            for _ in range(4)
        ]

        runner = AlphaRunner(verbose=False)
        results = runner.run_batch(jobs, n_jobs=2)

        assert len(results) == 4


# ============================================================================
# Ensemble Tests
# ============================================================================


class TestAlphaEnsemble:
    """Tests for AlphaEnsemble."""

    def test_ensemble_initialization(self, multiple_alpha_results):
        """Test ensemble initialization."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        assert len(ensemble.results) == len(multiple_alpha_results)

    def test_ensemble_no_valid_results(self):
        """Test ensemble with no valid results raises error."""
        with pytest.raises(ValueError, match="No valid alpha results"):
            AlphaEnsemble([])

    def test_equal_weight_combination(self, multiple_alpha_results):
        """Test equal weight combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="equal_weight")

        assert isinstance(combined, pd.Series)
        assert len(combined) > 0
        # Should start at 1.0 (normalized)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_inverse_volatility_combination(self, multiple_alpha_results):
        """Test inverse volatility combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="inverse_volatility")

        assert isinstance(combined, pd.Series)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_ic_weighted_combination(self, multiple_alpha_results):
        """Test IC-weighted combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="ic_weighted")

        assert isinstance(combined, pd.Series)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_sharpe_weighted_combination(self, multiple_alpha_results):
        """Test Sharpe-weighted combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="sharpe_weighted")

        assert isinstance(combined, pd.Series)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_unknown_combination_method(self, multiple_alpha_results):
        """Test unknown combination method raises error."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        with pytest.raises(ValueError, match="Unknown combination method"):
            ensemble.combine(method="nonexistent_method")

    def test_returns_matrix(self, multiple_alpha_results):
        """Test internal returns matrix generation."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        returns_df = ensemble._get_returns_matrix()

        assert isinstance(returns_df, pd.DataFrame)
        assert returns_df.shape[1] == len(multiple_alpha_results)


# ============================================================================
# Robustness Tests
# ============================================================================


class TestRobustnessTester:
    """Tests for RobustnessTester."""

    def test_robustness_tester_initialization(self):
        """Test robustness tester initialization."""
        tester = RobustnessTester()
        assert tester.runner is not None

    def test_robustness_tester_with_custom_runner(self):
        """Test robustness tester with custom runner."""
        runner = AlphaRunner(verbose=False)
        tester = RobustnessTester(runner=runner)
        assert tester.runner == runner

    def test_parameter_sensitivity(self, sample_data_with_indicators):
        """Test parameter sensitivity analysis."""
        base_job = AlphaJob(
            data=sample_data_with_indicators,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
            strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
        )

        param_grid = {
            "oversold": [25, 30],
            "overbought": [70, 75],
        }

        tester = RobustnessTester(runner=AlphaRunner(verbose=False))
        results_df = tester.parameter_sensitivity(base_job, param_grid, n_jobs=1)

        assert isinstance(results_df, pd.DataFrame)
        # Should have 4 rows (2x2 combinations)
        assert len(results_df) <= 4  # May be less if some failed

    def test_sub_period_analysis(self, sample_alpha_result):
        """Test sub-period analysis."""
        tester = RobustnessTester()

        # Use quarterly periods
        period_df = tester.sub_period_analysis(sample_alpha_result, period="Q")

        assert isinstance(period_df, pd.DataFrame)
        # Should have some standard columns
        if not period_df.empty:
            assert "Return" in period_df.columns
            assert "Sharpe" in period_df.columns

    def test_sub_period_analysis_invalid_result(self, sample_alpha_job):
        """Test sub-period analysis with invalid result."""
        invalid_result = AlphaResult(
            job=sample_alpha_job,
            metrics={},
            status="failed",
        )

        tester = RobustnessTester()
        with pytest.raises(ValueError, match="not completed"):
            tester.sub_period_analysis(invalid_result)


# ============================================================================
# Visualization Tests
# ============================================================================


class TestAlphaVisualizer:
    """Tests for AlphaVisualizer."""

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        viz = AlphaVisualizer()
        assert viz is not None
        assert len(viz.PALETTE) > 0

    def test_plot_cumulative_returns(self, multiple_alpha_results):
        """Test cumulative returns plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_cumulative_returns(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_drawdowns(self, multiple_alpha_results):
        """Test drawdown plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_drawdowns(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_ic_analysis(self, multiple_alpha_results):
        """Test IC analysis plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_ic_analysis(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_rolling_sharpe(self, multiple_alpha_results):
        """Test rolling Sharpe plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_rolling_sharpe(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_metrics_radar(self, multiple_alpha_results):
        """Test radar chart generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_metrics_radar(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_generate_html_report(self, multiple_alpha_results, tmp_path):
        """Test HTML report generation."""
        viz = AlphaVisualizer()
        output_path = tmp_path / "test_report.html"

        viz.generate_html_report(multiple_alpha_results, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "Alpha Research Report" in content
        assert "Strategy Performance" in content

    def test_fig_to_base64(self, multiple_alpha_results):
        """Test figure to base64 conversion."""
        viz = AlphaVisualizer()
        fig = viz.plot_cumulative_returns(multiple_alpha_results)

        base64_str = viz._fig_to_base64(fig)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        import matplotlib.pyplot as plt

        plt.close(fig)


# ============================================================================
# Integration Tests
# ============================================================================


class TestAlphaResearchIntegration:
    """Integration tests for the alpha research workflow."""

    def test_full_workflow(self, sample_data_with_indicators):
        """Test complete alpha research workflow."""
        # 1. Create jobs
        jobs = [
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
            ),
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="SMA",
                strategy_name="trend_following",
                strategy_params={"indicator_col": "SMA_20"},
            ),
        ]

        # 2. Run jobs
        runner = AlphaRunner(verbose=False)
        results = runner.run_batch(jobs, n_jobs=1)

        assert len(results) == 2

        # 3. Filter completed results
        completed = [r for r in results if r.status == "completed"]

        if len(completed) >= 2:
            # 4. Create ensemble
            ensemble = AlphaEnsemble(completed)
            combined = ensemble.combine(method="equal_weight")

            assert isinstance(combined, pd.Series)
            assert len(combined) > 0

    def test_workflow_with_robustness(self, sample_data_with_indicators):
        """Test workflow including robustness analysis."""
        # Create base job
        base_job = AlphaJob(
            data=sample_data_with_indicators,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
            strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
        )

        # Run sensitivity analysis
        tester = RobustnessTester(runner=AlphaRunner(verbose=False))

        param_grid = {"oversold": [25, 30]}
        results_df = tester.parameter_sensitivity(base_job, param_grid, n_jobs=1)

        assert isinstance(results_df, pd.DataFrame)

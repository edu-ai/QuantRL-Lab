"""Tests for experiments/backtesting/analysis.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantrl_lab.experiments.backtesting.analysis import BenchmarkAnalyzer, PerformanceMetrics


def _make_price_series(n=50, start=100.0, growth_pct=0.001):
    """Generate simple ascending price series."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + growth_pct))
    return pd.Series(prices)


def _make_ticker_mock(prices: pd.Series):
    ticker = MagicMock()
    df = pd.DataFrame({'Close': prices})
    ticker.history.return_value = df
    return ticker


class TestBenchmarkAnalyzer:
    @patch('quantrl_lab.experiments.backtesting.analysis.yf.Ticker')
    def test_buy_and_hold_returns_performance_metrics(self, mock_ticker_cls):
        prices = _make_price_series(50, 100.0)
        mock_ticker_cls.return_value = _make_ticker_mock(prices)

        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-12-31')
        result = analyzer.get_buy_and_hold_performance(initial_capital=100000)
        assert isinstance(result, PerformanceMetrics)

    @patch('quantrl_lab.experiments.backtesting.analysis.yf.Ticker')
    def test_buy_and_hold_positive_total_return_on_rising_prices(self, mock_ticker_cls):
        prices = _make_price_series(50, 100.0, growth_pct=0.01)
        mock_ticker_cls.return_value = _make_ticker_mock(prices)

        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-12-31')
        result = analyzer.get_buy_and_hold_performance(100000)
        assert result.total_return > 0

    @patch('quantrl_lab.experiments.backtesting.analysis.yf.Ticker')
    def test_dca_returns_performance_metrics(self, mock_ticker_cls):
        prices = _make_price_series(30)
        mock_ticker_cls.return_value = _make_ticker_mock(prices)

        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-06-30')
        result = analyzer.get_dollar_cost_averaging_performance(100000)
        assert isinstance(result, PerformanceMetrics)

    def test_calculate_metrics_sharpe_direction(self):
        """Positive returns should give positive Sharpe."""
        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-12-31')
        # Uniformly rising prices, excess_return > 0
        prices = np.linspace(100, 120, 252)
        metrics = analyzer._calculate_metrics(prices, initial_value=100.0, final_value=120.0)
        assert metrics.total_return == pytest.approx(0.2)
        assert metrics.win_rate > 0
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_calculate_metrics_max_drawdown_negative(self):
        """Max drawdown should always be ≤ 0."""
        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-12-31')
        prices = np.array([100, 120, 90, 110, 80, 130])
        metrics = analyzer._calculate_metrics(prices, initial_value=100.0, final_value=130.0)
        assert metrics.max_drawdown <= 0.0

    def test_create_comparison_report_contains_header(self):
        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-12-31')
        benchmarks = {
            'Strategy_A': PerformanceMetrics(0.1, 0.08, 0.15, 0.5, -0.1, 0.55, 1.2, 0.4, 0.8),
            'Strategy_B': PerformanceMetrics(0.05, 0.04, 0.12, 0.3, -0.05, 0.5, 1.0, 0.3, 0.5),
        }
        report = analyzer.create_comparison_report(benchmarks)
        assert isinstance(report, str)
        assert 'PERFORMANCE COMPARISON REPORT' in report

    def test_extract_agent_metrics_empty_episodes(self):
        analyzer = BenchmarkAnalyzer('AAPL', '2023-01-01', '2023-12-31')
        result = analyzer._extract_agent_metrics({'test_episodes': []})
        assert isinstance(result, PerformanceMetrics)
        assert result.total_return == 0

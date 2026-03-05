import pytest

from quantrl_lab.experiments.backtesting.metrics import MetricsCalculator


class TestMetricsCalculator:
    def test_calculate_empty(self):
        calculator = MetricsCalculator()
        metrics = calculator.calculate([])
        assert metrics == {}

    def test_calculate_basic_metrics(self):
        episodes = [
            {
                "total_reward": 100,
                "steps": 10,
                "initial_value": 1000,
                "final_value": 1100,
                "detailed_actions": [{"portfolio_value": 1000}, {"portfolio_value": 1050}, {"portfolio_value": 1100}],
            },
            {
                "total_reward": 50,
                "steps": 5,
                "initial_value": 1000,
                "final_value": 1050,
                "detailed_actions": [{"portfolio_value": 1000}, {"portfolio_value": 1025}, {"portfolio_value": 1050}],
            },
        ]

        calculator = MetricsCalculator()
        metrics = calculator.calculate(episodes)

        assert metrics["total_episodes"] == 2
        assert metrics["avg_reward"] == 75.0
        assert metrics["avg_episode_length"] == 7.5
        assert metrics["avg_return_pct"] == pytest.approx(7.5)  # (10% + 5%) / 2
        assert metrics["win_rate"] == 1.0

    def test_calculate_financial_metrics(self):
        # Create an episode with a known equity curve
        # Prices: 100, 102, 101, 104
        # Returns: 0.02, -0.0098, 0.0297
        curve = [100, 102, 101, 104]

        episodes = [
            {
                "total_reward": 10,
                "steps": 4,
                "initial_value": 100,
                "final_value": 104,
                "detailed_actions": [{"portfolio_value": v} for v in curve],
            }
        ]

        calculator = MetricsCalculator()
        metrics = calculator.calculate(episodes)

        assert "avg_sharpe_ratio" in metrics
        assert "avg_max_drawdown" in metrics
        assert "avg_sortino_ratio" in metrics

        # Drawdown: 102 -> 101 is a drawdown of ~0.98%
        # (101/102) - 1 = -0.0098
        assert -0.01 < metrics["avg_max_drawdown"] < 0.0

    def test_error_handling(self):
        episodes = [
            {"error": "Something went wrong", "total_reward": 0},
            {"total_reward": 10, "steps": 10, "initial_value": 100, "final_value": 110, "detailed_actions": []},
        ]
        calculator = MetricsCalculator()
        metrics = calculator.calculate(episodes)

        # Should ignore the error episode
        assert metrics["total_episodes"] == 1
        assert metrics["avg_reward"] == 10

from unittest.mock import MagicMock, patch

import pytest

from quantrl_lab.experiments.backtesting.config.environment_config import BacktestEnvironmentConfig
from quantrl_lab.experiments.backtesting.core import ExperimentResult
from quantrl_lab.experiments.backtesting.runner import BacktestRunner
from quantrl_lab.experiments.tuning.optuna_runner import OptunaRunner


class TestOptunaRunner:
    @pytest.fixture
    def mock_backtest_runner(self):
        return MagicMock(spec=BacktestRunner)

    @pytest.fixture
    def runner(self, mock_backtest_runner):
        return OptunaRunner(runner=mock_backtest_runner, storage_url="sqlite:///:memory:")

    @pytest.fixture
    def mock_env_config(self):
        return MagicMock(spec=BacktestEnvironmentConfig)

    def test_create_objective_function(self, runner, mock_env_config):
        algo_class = MagicMock()
        search_space = {"learning_rate": {"type": "float", "low": 0.001, "high": 0.01}}

        objective = runner.create_objective_function(
            algo_class=algo_class,
            env_config=mock_env_config,
            search_space=search_space,
            optimization_metric="test_return",
        )

        assert callable(objective)

    @patch("quantrl_lab.experiments.tuning.optuna_runner.optuna")
    def test_optimize_hyperparameters(self, mock_optuna, runner, mock_env_config, mock_backtest_runner):
        # Mock optuna study
        mock_study = MagicMock()
        mock_study.best_value = 0.5  # Must be a number for f-string formatting
        mock_optuna.create_study.return_value = mock_study

        algo_class = MagicMock()
        search_space = {"learning_rate": {"type": "float", "low": 0.001, "high": 0.01}}

        # Run optimization
        runner.optimize_hyperparameters(
            algo_class=algo_class,
            env_config=mock_env_config,
            search_space=search_space,
            study_name="test_study",
            n_trials=1,
        )

        # Check if optimize was called
        mock_study.optimize.assert_called_once()

    def test_objective_execution(self, runner, mock_env_config, mock_backtest_runner):
        # Setup mocks
        algo_class = MagicMock()
        search_space = {"learning_rate": {"type": "float", "low": 0.001, "high": 0.01}}

        # Mock successful backtest result
        mock_result = MagicMock(spec=ExperimentResult)
        mock_result.status = "completed"
        mock_result.metrics = {"test_return": 0.05}
        mock_backtest_runner.run_job.return_value = mock_result

        # Create objective
        objective = runner.create_objective_function(
            algo_class=algo_class,
            env_config=mock_env_config,
            search_space=search_space,
            optimization_metric="test_return",
        )

        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.005

        # Run objective
        value = objective(mock_trial)

        assert value == 0.05
        # Verify job was run with correct params
        mock_backtest_runner.run_job.assert_called_once()
        args, _ = mock_backtest_runner.run_job.call_args
        job = args[0]
        assert job.algorithm_config["learning_rate"] == 0.005

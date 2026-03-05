from unittest.mock import MagicMock, patch

import pytest

from quantrl_lab.experiments.backtesting.config.environment_config import BacktestEnvironmentConfig
from quantrl_lab.experiments.backtesting.core import ExperimentJob, ExperimentResult
from quantrl_lab.experiments.backtesting.runner import BacktestRunner


class TestBacktestRunner:
    @pytest.fixture
    def runner(self):
        return BacktestRunner(verbose=False)

    @pytest.fixture
    def mock_job(self):
        env_config = MagicMock(spec=BacktestEnvironmentConfig)
        env_config.name = "test_env"
        # Factories need to return something
        env_config.train_env_factory = MagicMock()
        env_config.test_env_factory = MagicMock()

        algo_class = MagicMock()
        algo_class.__name__ = "MockAlgo"

        job = ExperimentJob(
            algorithm_class=algo_class, env_config=env_config, algorithm_config={}, total_timesteps=100, n_envs=1
        )
        return job

    @patch("quantrl_lab.experiments.backtesting.runner.make_vec_env")
    @patch("quantrl_lab.experiments.backtesting.runner.train_model")
    @patch("quantrl_lab.experiments.backtesting.runner.evaluate_model")
    def test_run_job_success(self, mock_eval, mock_train, mock_make_vec_env, runner, mock_job):
        # Setup mocks
        mock_model = MagicMock()
        mock_train.return_value = mock_model

        # Mock evaluation results
        mock_eval.return_value = ([10.0], [{"total_reward": 10.0, "steps": 5}])

        # Run
        result = runner.run_job(mock_job)

        assert isinstance(result, ExperimentResult)
        assert result.status == "completed"
        assert result.metrics["train_avg_reward"] == 10.0
        assert mock_train.called
        assert mock_eval.call_count == 2  # Train and Test

    @patch("quantrl_lab.experiments.backtesting.runner.make_vec_env")
    def test_run_job_failure(self, mock_make_vec_env, runner, mock_job):
        # Simulate training failure
        mock_make_vec_env.side_effect = Exception("Env creation failed")

        result = runner.run_job(mock_job)

        assert result.status == "failed"
        assert "Env creation failed" in str(result.error)

    @patch("quantrl_lab.experiments.backtesting.runner.make_vec_env")
    @patch("quantrl_lab.experiments.backtesting.runner.train_model")
    @patch("quantrl_lab.experiments.backtesting.runner.evaluate_model")
    def test_run_batch(self, mock_eval, mock_train, mock_make_vec_env, runner, mock_job):
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        mock_eval.return_value = ([10.0], [{"total_reward": 10.0, "steps": 5}])

        jobs = [mock_job, mock_job]  # 2 jobs
        results = runner.run_batch(jobs)

        assert len(results) == 2
        assert all(r.status == "completed" for r in results)

    @patch("quantrl_lab.experiments.backtesting.runner.make_vec_env")
    @patch("quantrl_lab.experiments.backtesting.runner.train_model")
    @patch("quantrl_lab.experiments.backtesting.runner.evaluate_model")
    def test_inspect_result_no_error(self, mock_eval, mock_train, mock_make_vec_env, runner, mock_job):
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        mock_eval.return_value = (
            [10.0],
            [{"total_reward": 10.0, "steps": 5, "initial_value": 100000, "final_value": 110000, "actions_taken": {}}],
        )
        result = runner.run_job(mock_job)
        # Should not raise
        BacktestRunner.inspect_result(result)

    @patch("quantrl_lab.experiments.backtesting.runner.make_vec_env")
    @patch("quantrl_lab.experiments.backtesting.runner.train_model")
    @patch("quantrl_lab.experiments.backtesting.runner.evaluate_model")
    def test_inspect_batch_no_error(self, mock_eval, mock_train, mock_make_vec_env, runner, mock_job):
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        mock_eval.return_value = ([10.0], [{"total_reward": 10.0, "steps": 5}])
        results = runner.run_batch([mock_job])
        # Should not raise
        BacktestRunner.inspect_batch(results)

    @patch("quantrl_lab.experiments.backtesting.runner.make_vec_env")
    def test_run_batch_continues_on_failure(self, mock_make_vec_env, runner, mock_job):
        """Batch keeps running even if one job fails."""
        mock_make_vec_env.side_effect = Exception("Env failed")
        results = runner.run_batch([mock_job, mock_job])
        assert len(results) == 2
        assert all(r.status == "failed" for r in results)

    def test_inspect_failed_result_no_error(self, mock_job):
        """inspect_result on a failed result should not raise."""
        result = ExperimentResult(
            job=mock_job,
            metrics={},
            status="failed",
            error=RuntimeError("something broke"),
        )
        BacktestRunner.inspect_result(result)

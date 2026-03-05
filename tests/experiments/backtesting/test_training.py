"""Tests for experiments/backtesting/training.py."""

from unittest.mock import MagicMock

from quantrl_lab.experiments.backtesting.training import train_model


class TestTrainModel:
    def _make_algo_class(self):
        algo_class = MagicMock()
        algo_class.__name__ = 'MockPPO'
        mock_model = MagicMock()
        algo_class.return_value = mock_model
        return algo_class, mock_model

    def test_returns_trained_model(self):
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()
        result = train_model(algo_class, env, total_timesteps=10, verbose=0)
        assert result is mock_model
        mock_model.learn.assert_called_once_with(total_timesteps=10)

    def test_with_dict_config(self):
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()
        config = {'learning_rate': 0.001}
        train_model(algo_class, env, config=config, total_timesteps=10, verbose=0)
        call_kwargs = algo_class.call_args[1]
        assert call_kwargs['learning_rate'] == 0.001

    def test_with_object_config(self):
        """Config passed as object with __dict__."""
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()

        class FakeConfig:
            def __init__(self):
                self.learning_rate = 0.0003
                self.n_steps = 64

        train_model(algo_class, env, config=FakeConfig(), total_timesteps=10, verbose=0)
        call_kwargs = algo_class.call_args[1]
        assert call_kwargs['learning_rate'] == 0.0003

    def test_no_config_uses_defaults(self):
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()
        train_model(algo_class, env, config=None, total_timesteps=10, verbose=0)
        call_kwargs = algo_class.call_args[1]
        # Only base params should be set: policy, env, verbose
        assert 'policy' in call_kwargs
        assert 'env' in call_kwargs

    def test_kwargs_override_config(self):
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()
        config = {'learning_rate': 0.001}
        train_model(algo_class, env, config=config, total_timesteps=10, verbose=0, learning_rate=0.01)
        call_kwargs = algo_class.call_args[1]
        # kwargs should override config
        assert call_kwargs['learning_rate'] == 0.01

    def test_suppress_logs_sets_verbose_zero(self):
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()
        train_model(algo_class, env, total_timesteps=10, verbose=1, suppress_logs=True)
        call_kwargs = algo_class.call_args[1]
        assert call_kwargs['verbose'] == 0

    def test_default_policy_is_mlp(self):
        algo_class, mock_model = self._make_algo_class()
        env = MagicMock()
        train_model(algo_class, env, total_timesteps=10, verbose=0)
        call_kwargs = algo_class.call_args[1]
        assert call_kwargs['policy'] == 'MlpPolicy'

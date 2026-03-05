"""Tests for experiments/backtesting/explainer.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from quantrl_lab.experiments.backtesting.explainer import AgentExplainer


def _make_base_env(feature_names=None):
    """Create a mock base gym environment with observation strategy."""
    if feature_names is None:
        feature_names = ["feat_a", "feat_b", "feat_c"]

    env = MagicMock()
    obs_strategy = MagicMock()
    obs_strategy.get_feature_names.return_value = feature_names
    env.observation_strategy = obs_strategy
    return env


def _make_vec_env(base_env, n_obs_features=3, n_steps=5):
    """Mock vectorized env that runs for n_steps then terminates."""
    from stable_baselines3.common.vec_env import VecEnv

    vec_env = MagicMock(spec=VecEnv)
    vec_env.num_envs = 1

    obs = np.zeros((1, n_obs_features), dtype=np.float32)
    vec_env.reset.return_value = obs

    step_count = {"n": 0}

    def step(actions):
        step_count["n"] += 1
        done = step_count["n"] >= n_steps
        return (
            np.zeros((1, n_obs_features), dtype=np.float32),
            np.array([1.0]),
            np.array([done]),
            [{}],
        )

    vec_env.step.side_effect = step
    vec_env.envs = [base_env]
    return vec_env


def _make_model(n_features=3):
    model = MagicMock()
    model.__class__.__name__ = "PPO"
    model.predict.return_value = (np.array([0]), None)
    model.device = "cpu"

    # Setup policy for saliency (but we'll use correlation path in most tests)
    policy = MagicMock()
    model.policy = policy
    return model


class TestAgentExplainer:
    def test_extract_base_env_from_vec_env(self):
        base_env = MagicMock()
        vec_env = MagicMock()
        vec_env.envs = [base_env]
        model = MagicMock()
        explainer = AgentExplainer(model, vec_env)
        result = explainer.extract_base_env()
        assert result is base_env

    def test_extract_base_env_from_unwrapped(self):
        inner = MagicMock()
        env = MagicMock(spec=[])
        env.unwrapped = inner
        model = MagicMock()
        explainer = AgentExplainer(model, env)
        result = explainer.extract_base_env()
        assert result is inner

    def test_extract_base_env_direct(self):
        env = MagicMock(spec=[])  # no .envs, no .unwrapped
        model = MagicMock()
        explainer = AgentExplainer(model, env)
        result = explainer.extract_base_env()
        assert result is env

    def test_feature_importance_correlation_method(self):
        """Correlation method returns dict with feature scores."""
        feature_names = ["feat_a", "feat_b", "feat_c"]
        base_env = _make_base_env(feature_names)
        vec_env = _make_vec_env(base_env, n_obs_features=3, n_steps=10)
        model = _make_model(n_features=3)

        # Make model.predict return varying actions to allow correlation
        call_count = {"n": 0}
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        def predict(obs, **kwargs):
            idx = min(call_count["n"], len(actions) - 1)
            call_count["n"] += 1
            return (np.array([actions[idx]]), None)

        model.predict.side_effect = predict

        explainer = AgentExplainer(model, vec_env)
        result = explainer.analyze_feature_importance(top_k=3, method="correlation")
        assert isinstance(result, dict)
        assert len(result) <= 3

    def test_no_observation_strategy_raises(self):
        """Env without observation_strategy.get_feature_names raises
        NotImplementedError."""
        env = MagicMock()
        del env.observation_strategy  # Remove attribute
        vec_env = MagicMock()
        vec_env.num_envs = 1
        vec_env.envs = [env]

        model = MagicMock()
        model.__class__.__name__ = "PPO"
        explainer = AgentExplainer(model, vec_env)

        with pytest.raises(NotImplementedError):
            explainer.analyze_feature_importance(method="correlation")

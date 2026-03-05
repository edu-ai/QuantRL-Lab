"""Tests for experiments/backtesting/evaluation.py."""

from unittest.mock import MagicMock

import pytest

from quantrl_lab.experiments.backtesting.evaluation import (
    evaluate_model,
    evaluate_multiple_models,
    get_action_statistics,
)


def _make_mock_env(n_steps=3):
    """Create a mock gym environment that terminates after n_steps."""
    env = MagicMock()
    env.reset.return_value = ([0.0] * 5, {"portfolio_value": 100000.0})

    step_count = {"n": 0}

    def step(action):
        step_count["n"] += 1
        terminated = step_count["n"] >= n_steps
        truncated = False
        obs = [0.0] * 5
        reward = 1.0
        info = {
            "portfolio_value": 100000.0 + step_count["n"] * 100,
            "current_price": 150.0,
            "action_decoded": {
                "type": "hold",
                "amount_pct": 0.0,
                "price_modifier": 1.0,
                "invalid_action_attempt": False,
            },
        }
        return obs, reward, terminated, truncated, info

    env.step.side_effect = step
    return env


class TestEvaluateModel:
    def test_returns_rewards_and_results(self):
        model = MagicMock()
        model.__class__.__name__ = "MockPPO"
        model.predict.return_value = ([0], None)
        env = _make_mock_env(n_steps=2)

        rewards, results = evaluate_model(model, env, num_episodes=1, verbose=False)
        assert isinstance(rewards, list)
        assert isinstance(results, list)
        assert len(rewards) == 1
        assert len(results) == 1

    def test_episode_count(self):
        model = MagicMock()
        model.__class__.__name__ = "MockPPO"
        model.predict.return_value = ([0], None)

        rewards, results = evaluate_model(model, _make_mock_env(2), num_episodes=3, verbose=False)
        assert len(rewards) == 3
        assert len(results) == 3

    def test_accumulates_rewards(self):
        model = MagicMock()
        model.__class__.__name__ = "MockPPO"
        model.predict.return_value = ([0], None)
        env = _make_mock_env(n_steps=3)

        rewards, _ = evaluate_model(model, env, num_episodes=1, verbose=False)
        # 3 steps with reward=1.0 each
        assert rewards[0] == pytest.approx(3.0)

    def test_verbose_false_no_error(self):
        model = MagicMock()
        model.__class__.__name__ = "MockPPO"
        model.predict.return_value = ([0], None)
        env = _make_mock_env(2)
        # Should not raise
        evaluate_model(model, env, num_episodes=1, verbose=False)

    def test_result_contains_episode_number(self):
        model = MagicMock()
        model.__class__.__name__ = "MockPPO"
        model.predict.return_value = ([0], None)
        env = _make_mock_env(2)
        _, results = evaluate_model(model, env, num_episodes=2, verbose=False)
        assert results[0]["episode"] == 1
        assert results[1]["episode"] == 2


class TestEvaluateMultipleModels:
    def test_returns_dict_keyed_by_name(self):
        model1 = MagicMock()
        model1.__class__.__name__ = "PPO"
        model1.predict.return_value = ([0], None)

        model2 = MagicMock()
        model2.__class__.__name__ = "SAC"
        model2.predict.return_value = ([0], None)

        env = _make_mock_env(2)
        results = evaluate_multiple_models(
            models={"ppo": model1, "sac": model2},
            env=env,
            num_episodes=1,
            verbose=False,
        )
        assert "ppo" in results
        assert "sac" in results
        assert isinstance(results["ppo"], tuple)


class TestGetActionStatistics:
    def test_empty_results(self):
        stats = get_action_statistics([])
        assert stats["total_steps"] == 0
        assert stats["action_counts"] == {}

    def test_aggregates_actions(self):
        episodes = [
            {"steps": 5, "actions_taken": {"hold": 3, "buy": 2}},
            {"steps": 4, "actions_taken": {"hold": 2, "sell": 2}},
        ]
        stats = get_action_statistics(episodes)
        assert stats["total_steps"] == 9
        assert stats["action_counts"]["hold"] == 5
        assert stats["action_counts"]["buy"] == 2

    def test_percentages_sum_to_100(self):
        episodes = [{"steps": 10, "actions_taken": {"hold": 7, "buy": 3}}]
        stats = get_action_statistics(episodes)
        pcts = list(stats["action_percentages"].values())
        assert sum(pcts) == pytest.approx(100.0)

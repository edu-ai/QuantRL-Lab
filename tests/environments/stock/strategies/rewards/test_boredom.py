from unittest.mock import MagicMock

import pytest

from quantrl_lab.environments.stock.strategies.rewards.boredom import BoredomPenaltyReward


@pytest.fixture
def mock_portfolio():
    portfolio = MagicMock()
    portfolio.total_shares = 0
    portfolio.get_value.return_value = 10000.0
    return portfolio


@pytest.fixture
def mock_env(mock_portfolio):
    env = MagicMock()
    env.portfolio = mock_portfolio
    env._get_current_price.return_value = 100.0
    return env


def test_boredom_penalty_no_position(mock_env):
    """Test that reward is 0 when not holding any position."""
    reward_strat = BoredomPenaltyReward(penalty_per_step=-0.1, grace_period=5)
    mock_env.portfolio.total_shares = 0

    reward = reward_strat.calculate_reward(mock_env)
    assert reward == 0.0
    assert reward_strat._steps_held == 0


def test_boredom_penalty_grace_period(mock_env):
    """Test that no penalty is applied during the grace period."""
    reward_strat = BoredomPenaltyReward(penalty_per_step=-0.1, grace_period=5)
    mock_env.portfolio.total_shares = 10

    # Step 1-5: Inside grace period
    for _ in range(5):
        reward = reward_strat.calculate_reward(mock_env)
        assert reward == 0.0

    assert reward_strat._steps_held == 5


def test_boredom_penalty_active(mock_env):
    """Test that penalty is applied after grace period."""
    reward_strat = BoredomPenaltyReward(penalty_per_step=-0.1, grace_period=2)
    mock_env.portfolio.total_shares = 10

    # Step 1: Grace
    reward_strat.calculate_reward(mock_env)
    # Step 2: Grace
    reward_strat.calculate_reward(mock_env)

    # Step 3: Penalty!
    reward = reward_strat.calculate_reward(mock_env)
    assert reward == -0.1
    assert reward_strat._steps_held == 3


def test_boredom_reset_on_close(mock_env):
    """Test that counter resets when position is closed."""
    reward_strat = BoredomPenaltyReward(penalty_per_step=-0.1, grace_period=2)
    mock_env.portfolio.total_shares = 10

    # Hold for 3 steps (incur penalty)
    reward_strat.calculate_reward(mock_env)
    reward_strat.calculate_reward(mock_env)
    reward_strat.calculate_reward(mock_env)
    assert reward_strat._steps_held == 3

    # Close position
    mock_env.portfolio.total_shares = 0
    reward_strat.on_step_end(mock_env)

    assert reward_strat._steps_held == 0

    # Next step: Should be 0 (no position)
    reward = reward_strat.calculate_reward(mock_env)
    assert reward == 0.0

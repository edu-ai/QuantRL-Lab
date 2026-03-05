from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class DifferentialSharpeReward(BaseRewardStrategy):
    """
    Reward strategy based on the Differential Sharpe Ratio.

    Provides a dense reward signal at each step, representing the
    contribution of the current return to the overall Sharpe Ratio.

    It rewards high returns and penalizes total volatility.
    """

    def __init__(self, risk_free_rate: float = 0.0, decay: float = 0.99):
        """
        Args:
            risk_free_rate: The risk-free rate (per step) to subtract from returns.
                            Defaults to 0 assuming short time steps.
            decay: Decay factor for the moving average of returns and variance.
                   0 < decay < 1.
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.decay = decay

        # Moving statistics
        self._mean_return = 0.0
        self._mean_sq_return = 0.0  # E[x^2]
        self._step_count = 0

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """Calculate the differential Sharpe reward."""
        # Calculate current step return
        current_price = env._get_current_price()
        current_val = env.portfolio.get_value(current_price)
        prev_val = env.prev_portfolio_value

        if prev_val <= 1e-9:
            ret = 0.0
        else:
            ret = (current_val - prev_val) / prev_val

        # Excess return
        excess_ret = ret - self.risk_free_rate

        # Update moving averages using exponential decay
        if self._step_count == 0:
            self._mean_return = excess_ret
            self._mean_sq_return = excess_ret**2
        else:
            dt = 1.0 - self.decay
            self._mean_return = (1 - dt) * self._mean_return + dt * excess_ret
            self._mean_sq_return = (1 - dt) * self._mean_sq_return + dt * (excess_ret**2)

        self._step_count += 1

        # Calculate Variance: E[x^2] - (E[x])^2
        variance = self._mean_sq_return - (self._mean_return**2)
        # Ensure variance is non-negative (can be slightly negative due to float precision)
        variance = max(0.0, variance)

        std_dev = np.sqrt(variance) + 1e-9

        # A stable proxy for Differential Sharpe in RL context:
        # Reward = Excess_Return / Moving_Std_Dev
        # This scales the current return by the historical volatility environment.
        # If volatility is high, large returns are needed to get the same reward.

        return excess_ret / std_dev

    def on_step_end(self, env: TradingEnvProtocol):
        pass

    def reset(self):
        """Reset internal statistics."""
        self._mean_return = 0.0
        self._mean_sq_return = 0.0
        self._step_count = 0

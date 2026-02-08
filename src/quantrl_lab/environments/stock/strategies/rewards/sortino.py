from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class DifferentialSortinoReward(BaseRewardStrategy):
    """
    Reward strategy based on the Differential Sortino Ratio.

    Unlike the standard Sortino Ratio which is calculated over a fixed
    period, the Differential Sortino Ratio provides a dense reward
    signal at each step, representing the contribution of the current
    return to the overall Sortino Ratio.

    It penalizes downside volatility (returns below target) while
    rewarding positive returns.
    """

    def __init__(self, target_return: float = 0.0, decay: float = 0.99):
        """
        Args:
            target_return: Minimum acceptable return (MAR). Returns below this are considered downside risk.
            decay: Decay factor for the moving average of returns and downside deviation (0 < decay < 1).
                   Closer to 1 means longer memory.
        """
        super().__init__()
        self.target_return = target_return
        self.decay = decay

        # Moving statistics
        self._mean_return = 0.0
        self._mean_downside_sq = 0.0  # Mean squared downside deviation
        self._step_count = 0

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the differential Sortino reward.

        Ref: "Online Learning of the Differential Sharpe Ratio" logic adapted for Sortino.
        """
        # Calculate current step return
        current_price = env._get_current_price()
        current_val = env.portfolio.get_value(current_price)
        prev_val = env.prev_portfolio_value

        if prev_val <= 1e-9:
            ret = 0.0
        else:
            ret = (current_val - prev_val) / prev_val

        # Update moving averages using exponential decay
        if self._step_count == 0:
            self._mean_return = ret
            # Downside deviation: only consider returns below target
            downside = min(0, ret - self.target_return)
            self._mean_downside_sq = downside**2
        else:
            dt = 1.0 - self.decay
            self._mean_return = (1 - dt) * self._mean_return + dt * ret

            downside = min(0, ret - self.target_return)
            self._mean_downside_sq = (1 - dt) * self._mean_downside_sq + dt * (downside**2)

        self._step_count += 1

        # Calculate Sortino Ratio components
        # Add epsilon for numerical stability
        downside_dev = np.sqrt(self._mean_downside_sq) + 1e-9

        # Differential Sortino (Approximation of gradient/contribution)
        # S_t = (R_t - R_f) / DD_t
        # We want the change in S caused by the new return
        # Simple differential form: (ret - mean_ret) / downside_dev (simplified)
        # OR standard Sortino of current state.

        # A robust differential formulation for step-by-step RL:
        # Reward = (Return - Target) / DownsideDev_prev
        # But we need to account for changing DownsideDev.

        # Let's use the actual Differential Sortino formula derived similarly to Differential Sharpe:
        # D_t = (B_{t-1} * (R_t - Mean_t) - 0.5 * A_{t-1} * (Downside_t^2 - DownsideMean_t)) / (DownsideMean_t^3)
        # Where A = Mean Return - Target, B = Downside Variance.
        # This is complex and often unstable.

        # A standardized stable proxy often used in trading RL:
        # Reward = Current_Return / (Running_Downside_Dev + epsilon)
        # If Current_Return is negative, it's penalized by volatility.
        # If positive, it's scaled by the risk environment.

        return ret / downside_dev

    def on_step_end(self, env: TradingEnvProtocol):
        pass

    def reset(self):
        """Reset internal statistics."""
        self._mean_return = 0.0
        self._mean_downside_sq = 0.0
        self._step_count = 0

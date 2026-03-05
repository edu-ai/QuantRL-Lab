from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class DrawdownPenaltyReward(BaseRewardStrategy):
    """
    Penalizes the agent proportional to the current drawdown depth.

    This provides a continuous pressure to recover from losses.
    Reward = - (Current_Drawdown_Pct * penalty_factor)
    """

    def __init__(self, penalty_factor: float = 1.0):
        """
        Args:
            penalty_factor: Scaling factor for the penalty.
        """
        super().__init__()
        self.penalty_factor = penalty_factor
        self._max_portfolio_value = 0.0

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """Calculate drawdown penalty."""
        current_price = env._get_current_price()
        current_val = env.portfolio.get_value(current_price)

        # Initialize max value if first step (or if portfolio was reset but strategy wasn't)
        if self._max_portfolio_value == 0.0:
            self._max_portfolio_value = current_val
            # Avoid division by zero if starting with 0 balance (unlikely)
            if self._max_portfolio_value <= 1e-9:
                self._max_portfolio_value = 1e-9

        # Update high-water mark
        if current_val > self._max_portfolio_value:
            self._max_portfolio_value = current_val

        # Calculate drawdown
        drawdown_pct = (self._max_portfolio_value - current_val) / self._max_portfolio_value

        # Ensure drawdown is non-negative (it should be by definition, but float errors exist)
        drawdown_pct = max(0.0, drawdown_pct)

        return -(drawdown_pct * self.penalty_factor)

    def on_step_end(self, env: TradingEnvProtocol):
        pass

    def reset(self):
        """Reset internal high-water mark."""
        self._max_portfolio_value = 0.0

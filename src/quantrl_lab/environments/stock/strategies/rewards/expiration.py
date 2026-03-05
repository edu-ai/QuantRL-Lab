from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class OrderExpirationPenaltyReward(BaseRewardStrategy):
    """
    Penalizes the agent when pending orders expire.

    This discourages "order spamming" (placing unrealistic limit orders
    that never fill and just clog the system until they time out).
    """

    def __init__(self, penalty_per_order: float = -0.1):
        """
        Args:
            penalty_per_order: Fixed penalty for each expired order in the step.
                               Should be small but non-zero.
        """
        self.penalty_per_order = penalty_per_order

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """Calculate penalty based on number of expired orders in this
        step."""
        expired_count = 0

        if env.portfolio.executed_orders_history:
            for event in reversed(env.portfolio.executed_orders_history):
                if event["step"] != env.current_step:
                    break

                # Check for expiration event types
                # StockPortfolio logs types like "limit_buy_expired", "stop_loss_expired"
                if "expired" in event["type"]:
                    expired_count += 1

        return expired_count * self.penalty_per_order

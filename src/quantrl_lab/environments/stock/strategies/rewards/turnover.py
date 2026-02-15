from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class TurnoverPenaltyReward(BaseRewardStrategy):
    """
    Penalizes excessive trading by applying a multiple of the fees paid.

    While PnL implicitly accounts for fees, an explicit penalty helps
    the agent learn "efficiency" faster, discouraging noise trading
    where the profit margin is razor-thin compared to the cost.
    """

    def __init__(self, penalty_factor: float = 1.0):
        """
        Args:
            penalty_factor: Multiplier for fees paid.
                            1.0 means penalty = fees (doubling the cost impact).
                            5.0 means extremely high penalty for churning.
        """
        self.penalty_factor = penalty_factor

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate penalty based on transaction costs incurred in this
        step.

        We look at the executed_orders_history for events that happened
        at the current step.
        """
        fees_paid = 0.0

        # Check if history exists
        if env.portfolio.executed_orders_history:
            # Iterate backwards to find orders from current step
            # Optimally, we could just look at the last few, but this is safe
            for event in reversed(env.portfolio.executed_orders_history):
                if event["step"] != env.current_step:
                    break

                # Logic for cost extraction depends on event type
                # Market Buy / Limit Buy Executed: 'cost' field is the total cash spent
                # We need to estimate the *fee* portion.
                # StockPortfolio doesn't log the raw fee explicitly in the event dict,
                # but we can infer it or we might need to update StockPortfolio to log 'fee'.

                # Current StockPortfolio logic:
                # Buy: cost = shares * price * (1 + fee_pct)
                # Sell: revenue = shares * price * (1 - fee_pct)

                # To be precise without modifying Portfolio, we can approximate:
                # fee ~= value * fee_pct

                price = event.get("price", 0.0)
                shares = event.get("shares", 0)
                value = price * shares

                # Use the environment's configured transaction cost
                # env.portfolio.transaction_cost_pct is available
                fees_paid += value * env.portfolio.transaction_cost_pct

        return -(fees_paid * self.penalty_factor)

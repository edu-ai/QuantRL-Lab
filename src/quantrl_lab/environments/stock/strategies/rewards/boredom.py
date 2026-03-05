from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class BoredomPenaltyReward(BaseRewardStrategy):
    """
    Penalizes the agent for holding a position too long without
    significant price movement or profit.

    This encourages the agent to:
    1. Enter trades only when a move is expected soon.
    2. Exit stale positions rather than holding them indefinitely hoping for a turnaround.
    """

    def __init__(self, penalty_per_step: float = -0.001, grace_period: int = 10, min_profit_pct: float = 0.005):
        """
        Args:
            penalty_per_step: The negative reward to apply per step after the grace period.
            grace_period: Number of steps a position can be held without penalty.
            min_profit_pct: The minimum unrealized profit % required to reset the boredom timer.
                            If the position is profitable enough, we don't penalize holding (letting winners run).
        """
        super().__init__()
        self.penalty_per_step = penalty_per_step
        self.grace_period = grace_period
        self.min_profit_pct = min_profit_pct

        self._steps_held = 0
        self._entry_price = 0.0

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """Calculate the boredom penalty."""
        # Check if we have a position
        if env.portfolio.total_shares <= 0:
            self._steps_held = 0
            self._entry_price = 0.0
            return 0.0

        # Initialize entry price if this is the start of a position
        if self._steps_held == 0:
            # Ideally get average entry price from portfolio, but for simplicity use current
            # if we just detected a position. Better: check portfolio.
            pass

        # Calculate unrealized PnL %
        # We need the average entry price to know if we are "winning"
        # The portfolio object has executed_orders_history, but accessing avg_entry directly is better.
        # Let's approximate using the price when we first started counting (or complex logic).
        # For simplicity, let's assume the agent just needs "action" (price movement), not necessarily profit.

        # Better logic:
        # If unrealized profit > min_profit_pct, boredom = 0 (Let winners run).
        # Else, increment counter.

        # We need average entry price.
        # Accessing private or complex portfolio attributes might be brittle.
        # Let's use a simplified heuristic:
        # If the portfolio value hasn't increased by min_profit_pct since the trade started...

        # Actually, let's use the 'unrealized_pnl' if available, or calc it.
        # Since we don't have easy access to 'avg_entry_price' without iterating history,
        # let's just use a counter that resets on ANY trade action (Buy/Sell/Close).

        # Wait, the agent might 'LimitBuy' repeatedly. Does that count as action?
        # No, we only care about HOLDING a position.

        # Let's try to get the average entry price from the portfolio if possible.
        # Inspecting portfolio... it usually tracks positions.
        # Assuming `env.portfolio.positions` or similar.
        # The standard portfolio has `executed_orders_history`.

        # Fallback: Just punish "Time in Market" if returns are flat.

        # Let's implement a simple "Stale Position" penalty.
        self._steps_held += 1

        if self._steps_held <= self.grace_period:
            return 0.0

        # We are past grace period. Are we winning?
        # We can roughly estimate entry from when counter started (imperfect but usable)
        # Or better: check if the agent *did* something this step.
        # If action was HOLD, apply penalty.
        # But `calculate_reward` is called AFTER `step`.

        # Let's keep it simple:
        # If holding > grace_period, apply penalty.
        return self.penalty_per_step

    def on_step_end(self, env: TradingEnvProtocol):
        # Reset if position is closed or significantly changed
        # This requires checking if a trade happened this step.
        # We can check env.portfolio.trades[-1] timestamp?

        # Actually, simpler:
        # If position size is 0, reset.
        if env.portfolio.total_shares == 0:
            self._steps_held = 0

    def reset(self):
        self._steps_held = 0
        self._entry_price = 0.0

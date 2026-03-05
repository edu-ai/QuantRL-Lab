from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.environments.core.interfaces import BaseRewardStrategy

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class LimitExecutionReward(BaseRewardStrategy):
    """Provides a reward proportional to the price improvement achieved
    by a Limit Order filling instead of executing immediately at
    market."""

    def __init__(self, improvement_multiplier: float = 10.0):
        """
        Args:
            improvement_multiplier: Scales the % improvement.
                                    e.g., a 2% price improvement * 10.0 = +0.20 reward.
        """
        super().__init__()
        self.improvement_multiplier = improvement_multiplier

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        bonus = 0.0

        if env.portfolio.executed_orders_history:
            for event in reversed(env.portfolio.executed_orders_history):
                if event.get("step") != env.current_step:
                    break

                order_type = event.get("type", "")
                exec_price = event.get("price", 0.0)
                ref_price = event.get("reference_price", 0.0)

                if ref_price <= 1e-9:
                    continue

                improvement_pct = 0.0
                if order_type == "limit_buy_executed":
                    # Bought cheaper than the reference price
                    improvement_pct = (ref_price - exec_price) / ref_price
                elif order_type == "limit_sell_executed":
                    # Sold higher than the reference price
                    improvement_pct = (exec_price - ref_price) / ref_price

                if improvement_pct > 0:
                    bonus += improvement_pct * self.improvement_multiplier

        return bonus

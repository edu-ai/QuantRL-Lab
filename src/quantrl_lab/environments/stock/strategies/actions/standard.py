from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from quantrl_lab.environments.core.interfaces import (
    BaseActionStrategy,
)
from quantrl_lab.environments.core.types import Actions

if TYPE_CHECKING:  # Solves circular import issues
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class StandardActionStrategy(BaseActionStrategy):
    """
    Implements the full-featured action space with a 3-part Box space.

    Action: [action_type, amount, price_modifier]
    """

    def define_action_space(self) -> gym.spaces.Box:
        """
        Defines the action space for the trading environment.

        Returns:
            gym.spaces.Box: The action space as a Box space.
        """
        # We use a symmetric action space [-1, 1] for the action type to help RL agents
        # explore more effectively. An uninitialized agent outputs values near 0.
        # If we used [0, N], 0 would map to Action 0 (Hold), causing inactivity.
        # With [-1, 1], 0 maps to the middle action, encouraging interaction.
        action_type_low = -1.0
        action_type_high = 1.0

        # Use symmetric space for amount as well to avoid 0.0 default
        amount_low = -1.0
        amount_high = 1.0

        # Price modifier for limit orders, typically between 0.9 and 1.1
        price_mod_low = 0.9
        price_mod_high = 1.1

        return gym.spaces.Box(
            low=np.array([action_type_low, amount_low, price_mod_low], dtype=np.float32),
            high=np.array([action_type_high, amount_high, price_mod_high], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def handle_action(self, env_self: TradingEnvProtocol, action: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """
        Handles the action by decoding it and instructing the
        environment's portfolio.

        Args:
            env_self (TradingEnvProtocol): The environment instance.
            action (np.ndarray): The raw action from the agent.

        Returns:
            Tuple[Any, Dict[str, Any]]: The decoded action type and a dictionary of details.
        """
        # --- 1. Decode the action ---

        # Rescale Action Type from [-1, 1] to [0, len(Actions)-1]
        raw_type = np.clip(action[0], -1.0, 1.0)
        max_action_index = len(Actions) - 1
        scaled_type = ((raw_type + 1) / 2) * max_action_index

        action_type_int = int(np.round(scaled_type))

        # Rescale Amount from [-1, 1] to [0, 1]
        # This ensures that an output of 0.0 results in 50% amount, not 0%.
        raw_amount = np.clip(action[1], -1.0, 1.0)
        amount_pct = (raw_amount + 1) / 2

        price_modifier = np.clip(action[2], 0.9, 1.1)

        try:
            action_type = Actions(action_type_int)
        except ValueError:
            action_type = Actions.Hold

        # The environment is still responsible for providing the current price
        current_price = env_self._get_current_price()
        if current_price <= 1e-9:
            action_type = Actions.Hold

        # --- 2. Execute the action by calling methods on the PORTFOLIO ---

        # CORRECTED: Get total shares from the portfolio
        had_no_shares = env_self.portfolio.total_shares <= 0
        invalid_action_attempt = False

        # Get current_step from the environment, as the portfolio methods need it
        current_step = env_self.current_step

        if action_type == Actions.Hold:
            pass
        elif action_type == Actions.Buy:
            # CORRECTED: Call the portfolio's method, passing current_step
            env_self.portfolio.execute_market_order(action_type, current_price, amount_pct, current_step)
        elif action_type == Actions.Sell:
            if had_no_shares:
                invalid_action_attempt = True
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.execute_market_order(action_type, current_price, amount_pct, current_step)
        elif action_type == Actions.LimitBuy:
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.place_limit_order(action_type, current_price, amount_pct, price_modifier, current_step)
        elif action_type == Actions.LimitSell:
            if had_no_shares:
                invalid_action_attempt = True
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.place_limit_order(action_type, current_price, amount_pct, price_modifier, current_step)
        elif action_type in [Actions.StopLoss, Actions.TakeProfit]:
            if had_no_shares:
                invalid_action_attempt = True
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.place_risk_management_order(
                action_type, current_price, amount_pct, price_modifier, current_step
            )

        # --- 3. Return decoded info (No changes needed here) ---
        decoded_info = {
            "type": action_type.name,
            "amount_pct": amount_pct,
            "price_modifier": price_modifier,
            "raw_input": action,
            "invalid_action_attempt": invalid_action_attempt,
        }

        return action_type, decoded_info

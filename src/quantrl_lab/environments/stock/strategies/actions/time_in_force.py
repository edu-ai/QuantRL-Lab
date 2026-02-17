from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from quantrl_lab.environments.core.interfaces import (
    BaseActionStrategy,
)
from quantrl_lab.environments.core.types import Actions
from quantrl_lab.environments.stock.components.portfolio import OrderTIF

if TYPE_CHECKING:  # Solves circular import issues
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class TimeInForceActionStrategy(BaseActionStrategy):
    """
    Implements an advanced action space with Time-In-Force (TIF)
    control.

    Action: [action_type, amount, price_modifier, tif_type]

    TIF Types:
    0: GTC (Good Till Cancelled)
    1: IOC (Immediate or Cancel)
    2: TTL (Time To Live - uses order_expiration_steps)
    """

    def define_action_space(self) -> gym.spaces.Box:
        """
        Defines the action space for the trading environment.

        Returns:
            gym.spaces.Box: The action space as a Box space.
        """
        # Symmetric action space [-1, 1] for categorical actions to aid exploration

        # Action Type
        action_type_low = -1.0
        action_type_high = 1.0

        # Amount
        amount_low = 0.0
        amount_high = 1.0

        # Price Modifier
        price_mod_low = 0.9
        price_mod_high = 1.1

        # TIF Type (0: GTC, 1: IOC, 2: TTL)
        # Using symmetric space [-1, 1]
        tif_low = -1.0
        tif_high = 1.0

        return gym.spaces.Box(
            low=np.array([action_type_low, amount_low, price_mod_low, tif_low], dtype=np.float32),
            high=np.array([action_type_high, amount_high, price_mod_high, tif_high], dtype=np.float32),
            shape=(4,),
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
        raw_action_type = np.clip(action[0], -1.0, 1.0)
        max_action_index = len(Actions) - 1
        scaled_action_type = ((raw_action_type + 1) / 2) * max_action_index
        action_type_int = int(np.round(scaled_action_type))

        amount_pct = np.clip(action[1], 0.0, 1.0)
        price_modifier = np.clip(action[2], 0.9, 1.1)

        # Rescale TIF from [-1, 1] to [0, 2]
        raw_tif = np.clip(action[3], -1.0, 1.0)
        max_tif_index = 2
        scaled_tif = ((raw_tif + 1) / 2) * max_tif_index
        tif_int = int(np.round(scaled_tif))

        if tif_int == 0:
            tif_type = OrderTIF.GTC
        elif tif_int == 1:
            tif_type = OrderTIF.IOC
        else:
            tif_type = OrderTIF.TTL

        try:
            action_type = Actions(action_type_int)
        except ValueError:
            action_type = Actions.Hold

        # The environment is still responsible for providing the current price
        current_price = env_self._get_current_price()
        if current_price <= 1e-9:
            action_type = Actions.Hold

        # --- 2. Execute the action by calling methods on the PORTFOLIO ---

        had_no_shares = env_self.portfolio.total_shares <= 0
        invalid_action_attempt = False

        # Get current_step from the environment
        current_step = env_self.current_step

        if action_type == Actions.Hold:
            pass
        elif action_type == Actions.Buy:
            # Market orders don't use TIF (usually IOC by definition, handled internally)
            env_self.portfolio.execute_market_order(action_type, current_price, amount_pct, current_step)
        elif action_type == Actions.Sell:
            if had_no_shares:
                invalid_action_attempt = True
            env_self.portfolio.execute_market_order(action_type, current_price, amount_pct, current_step)
        elif action_type == Actions.LimitBuy:
            env_self.portfolio.place_limit_order(
                action_type, current_price, amount_pct, price_modifier, current_step, tif=tif_type
            )
        elif action_type == Actions.LimitSell:
            if had_no_shares:
                invalid_action_attempt = True
            env_self.portfolio.place_limit_order(
                action_type, current_price, amount_pct, price_modifier, current_step, tif=tif_type
            )
        elif action_type in [Actions.StopLoss, Actions.TakeProfit]:
            if had_no_shares:
                invalid_action_attempt = True

            # Stop orders cannot be IOC usually (they need to rest until triggered)
            # If IOC is selected for Stop, Portfolio handles it (likely rejects or ignores)
            # but we pass it anyway as the portfolio owns that validation logic.
            env_self.portfolio.place_risk_management_order(
                action_type, current_price, amount_pct, price_modifier, current_step, tif=tif_type
            )

        # --- 3. Return decoded info ---
        decoded_info = {
            "type": action_type.name,
            "amount_pct": amount_pct,
            "price_modifier": price_modifier,
            "tif": tif_type.value,
            "raw_input": action,
            "invalid_action_attempt": invalid_action_attempt,
        }

        return action_type, decoded_info

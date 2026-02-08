from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from quantrl_lab.environments.core.portfolio import Portfolio
from quantrl_lab.environments.core.types import Actions


class OrderType(str, Enum):
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderTIF(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    TTL = "TTL"  # Time To Live (uses config.order_expiration_steps)


@dataclass
class Order:
    type: OrderType
    shares: int
    price: float
    placed_at: int
    cost_reserved: float = 0.0  # Only for limit buy orders
    tif: OrderTIF = OrderTIF.GTC


class StockPortfolio(Portfolio):
    """
    A portfolio for stock trading that handles complex order types,
    fees, and slippage.

    It extends the simple Portfolio with stock-specific logic and state.
    """

    def __init__(
        self,
        initial_balance: float,
        transaction_cost_pct: float,
        slippage: float,
        order_expiration_steps: int,
    ):
        # === Initialize the parent class with the part it cares about ===
        super().__init__(initial_balance=initial_balance)

        # === Transaction cost and slippage can be adjusted to reflect difficulties in trading ===
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage = slippage
        self.order_expiration_steps = order_expiration_steps

        # === Stock-specific state ===
        # Using strict typing with dataclasses instead of generic dicts
        self.pending_orders: List[Order] = []
        self.stop_loss_orders: List[Order] = []
        self.take_profit_orders: List[Order] = []

        # We keep history as Dict for now to allow flexible logging and compatibility with existing renderers
        self.executed_orders_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset the portfolio to its initial state."""
        super().reset()
        self.pending_orders = []
        self.stop_loss_orders = []
        self.take_profit_orders = []
        self.executed_orders_history = []

    @property
    def shares_held(self) -> int:
        """
        Returns the number of shares currently held in the portfolio.

        Returns:
            int: The number of shares held.
        """
        return self.units_held

    @property
    def total_shares(self) -> int:
        """
        Returns the total number of shares held, including those
        reserved in orders.

        Returns:
            int: The total number of shares held.
        """
        return self.units_held + self._get_reserved_shares()

    def get_value(self, current_price: float) -> float:
        """
        Calculate the total value of the portfolio including unfilled
        orders and reserved money.

        Args:
            current_price (float): The current market price of the asset.

        Returns:
            float: The total portfolio value including all positions and reserved amounts.
        """
        # Base value: free balance + value of free shares
        total_value = self.balance + (self.units_held * current_price)

        # Add reserved cash from pending buy orders
        for order in self.pending_orders:
            if order.type == OrderType.LIMIT_BUY:
                total_value += order.cost_reserved

        # Add value of shares reserved in pending sell orders
        for order in self.pending_orders:
            if order.type == OrderType.LIMIT_SELL:
                total_value += order.shares * current_price

        # Add value of shares reserved in stop loss orders
        for order in self.stop_loss_orders:
            total_value += order.shares * current_price

        # Add value of shares reserved in take profit orders
        for order in self.take_profit_orders:
            total_value += order.shares * current_price

        return total_value

    def process_open_orders(
        self,
        current_step: int,
        current_price: float,
        current_high: Optional[float] = None,
        current_low: Optional[float] = None,
        current_open: Optional[float] = None,
    ) -> None:
        """
        Process all open orders using OHLC data for realistic execution.

        Args:
            current_step (int): The current step in the trading environment.
            current_price (float): The current close price.
            current_high (Optional[float]): High price of the bar. Defaults to current_price.
            current_low (Optional[float]): Low price of the bar. Defaults to current_price.
            current_open (Optional[float]): Open price of the bar. Defaults to current_price.
        """
        # Fallback for Close-only execution (backward compatibility)
        if current_high is None:
            current_high = current_price
        if current_low is None:
            current_low = current_price
        if current_open is None:
            current_open = current_price

        self._process_pending_orders(current_step, current_price, current_high, current_low, current_open)
        self._process_risk_management_orders(current_step, current_price, current_high, current_low, current_open)

    def execute_market_order(
        self, action_type: Actions, current_price: float, amount_pct: float, current_step: int
    ) -> None:
        """
        Execute a market order.

        Args:
            action_type (Actions): The type of action (buy/sell).
            current_price (float): The current market price.
            amount_pct (float): The percentage of the portfolio to use for the order.
            current_step (int): The current step in the trading environment.

        Returns:
            None
        """
        # Clip amount_pct to valid range
        amount_pct = max(0.0, min(1.0, amount_pct))

        # === Runtime error checks ===
        if self.balance <= 0 and action_type == Actions.Buy:
            raise ValueError("Insufficient balance to execute buy order")
        if action_type not in [Actions.Buy, Actions.Sell]:
            raise ValueError("Invalid action type for market order")

        # === Buy Logic ===
        if action_type == Actions.Buy:
            adjusted_price = current_price * (1 + self.slippage)
            cost_per_share = adjusted_price * (1 + self.transaction_cost_pct)
            if cost_per_share <= 1e-9:
                return  # Avoid division by zero

            shares_to_buy = int((self.balance / cost_per_share) * amount_pct)
            if shares_to_buy > 0:
                actual_cost = shares_to_buy * cost_per_share
                if actual_cost <= self.balance:
                    self.balance -= actual_cost
                    self.units_held += shares_to_buy
                    self.executed_orders_history.append(
                        {
                            "step": current_step,
                            "type": "market_buy",
                            "shares": shares_to_buy,
                            "price": adjusted_price,
                            "cost": actual_cost,
                        }
                    )

        # === Sell Logic ===
        elif action_type == Actions.Sell:
            if self.units_held <= 0:
                return
            shares_to_sell = int(self.units_held * amount_pct)
            if shares_to_sell > 0:
                adjusted_price = current_price * (1 - self.slippage)
                revenue = shares_to_sell * adjusted_price * (1 - self.transaction_cost_pct)
                self.units_held -= shares_to_sell
                self.balance += revenue
                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "market_sell",
                        "shares": shares_to_sell,
                        "price": adjusted_price,
                        "revenue": revenue,
                    }
                )

    def place_limit_order(
        self,
        action_type: Actions,
        current_price: float,
        amount_pct: float,
        price_modifier: float,
        current_step: int,
        tif: OrderTIF = OrderTIF.TTL,  # Default to TTL to preserve previous behavior
    ) -> None:
        """
        Place a limit order for buying or selling an asset.

        Args:
            action_type (Actions): The type of action (LimitBuy/LimitSell).
            current_price (float): The current market price.
            amount_pct (float): The percentage of the portfolio to use for the order.
            price_modifier (float): The price modifier to apply to the current price.
            current_step (int): The current step in the trading environment.
            tif (OrderTIF): Time in Force for the order.

        Returns:
            None
        """
        limit_price = current_price * price_modifier

        # === Limit Buy Logic ===
        if action_type == Actions.LimitBuy:
            cost_per_share = limit_price * (1 + self.transaction_cost_pct)
            if cost_per_share <= 1e-9:
                return
            shares_to_buy = int((self.balance / cost_per_share) * amount_pct)
            if shares_to_buy > 0:
                cost_reserved = shares_to_buy * cost_per_share

                # Check balance
                if cost_reserved > self.balance:
                    return

                # --- Handle IOC (Immediate or Cancel) ---
                if tif == OrderTIF.IOC:
                    # If current price <= limit price, execute immediately
                    if current_price <= limit_price:
                        # IOC Execution matches logic of standard execution
                        execution_price = limit_price  # or current_price? Standard logic uses limit_price

                        self.balance -= cost_reserved
                        # Add shares (execution success)
                        self.units_held += shares_to_buy

                        self.executed_orders_history.append(
                            {
                                "step": current_step,
                                "type": "limit_buy_executed_ioc",
                                "shares": shares_to_buy,
                                "price": execution_price,
                                "cost": cost_reserved,
                            }
                        )
                    # If not executable, do nothing (cancel)
                    return

                # --- Handle GTC / TTL (Pending) ---
                self.balance -= cost_reserved
                order = Order(
                    type=OrderType.LIMIT_BUY,
                    shares=shares_to_buy,
                    price=limit_price,
                    placed_at=current_step,
                    cost_reserved=cost_reserved,
                    tif=tif,
                )
                self.pending_orders.append(order)

                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "limit_buy_placed",
                        "shares": shares_to_buy,
                        "price": limit_price,
                        "tif": tif.value,
                    }
                )

        # === Limit Sell Logic ===
        elif action_type == Actions.LimitSell:
            if self.units_held <= 0:
                return
            shares_to_sell = int(self.units_held * amount_pct)
            if shares_to_sell > 0:

                # --- Handle IOC (Immediate or Cancel) ---
                if tif == OrderTIF.IOC:
                    # If current price >= limit price, execute immediately
                    if current_price >= limit_price:
                        execution_price = limit_price

                        # Calculate revenue
                        revenue = shares_to_sell * execution_price * (1 - self.transaction_cost_pct)

                        self.units_held -= shares_to_sell
                        self.balance += revenue

                        self.executed_orders_history.append(
                            {
                                "step": current_step,
                                "type": "limit_sell_executed_ioc",
                                "shares": shares_to_sell,
                                "price": execution_price,
                                "revenue": revenue,
                            }
                        )
                    # If not executable, do nothing (cancel)
                    return

                # --- Handle GTC / TTL (Pending) ---
                self.units_held -= shares_to_sell
                order = Order(
                    type=OrderType.LIMIT_SELL, shares=shares_to_sell, price=limit_price, placed_at=current_step, tif=tif
                )
                self.pending_orders.append(order)

                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "limit_sell_placed",
                        "shares": shares_to_sell,
                        "price": limit_price,
                        "tif": tif.value,
                    }
                )

    def place_risk_management_order(
        self,
        action_type: Actions,
        current_price: float,
        amount_pct: float,
        price_modifier: float,
        current_step: int,
        tif: OrderTIF = OrderTIF.GTC,  # Default to GTC (standard for stop loss)
    ) -> None:
        """
        Place a risk management order (stop loss or take profit).

        Args:
            action_type (Actions): The type of action (StopLoss/TakeProfit).
            current_price (float): The current market price.
            amount_pct (float): The percentage of the portfolio to use for the order.
            price_modifier (float): The price modifier to apply to the current price.
            current_step (int): The current step in the trading environment.
            tif (OrderTIF): Time in Force. Only GTC and TTL are valid for Stop orders.

        Returns:
            None
        """
        # Validate TIF for Stop orders
        if tif == OrderTIF.IOC:
            return  # IOC is invalid for Stop orders (must rest until trigger)

        if self.units_held <= 0:
            return
        shares_to_cover = int(self.units_held * amount_pct)
        if shares_to_cover > 0:
            # === Stop Loss Logic ===
            if action_type == Actions.StopLoss:
                stop_price = current_price * min(0.999, price_modifier)
                if stop_price >= current_price:
                    stop_price = current_price * 0.999

                self.units_held -= shares_to_cover

                order = Order(
                    type=OrderType.STOP_LOSS, shares=shares_to_cover, price=stop_price, placed_at=current_step, tif=tif
                )
                self.stop_loss_orders.append(order)

                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "stop_loss_placed",
                        "shares": shares_to_cover,
                        "price": stop_price,
                        "tif": tif.value,
                    }
                )
            # === Take Profit Logic ===
            elif action_type == Actions.TakeProfit:
                take_profit_price = current_price * max(1.001, price_modifier)
                if take_profit_price <= current_price:
                    take_profit_price = current_price * 1.001

                self.units_held -= shares_to_cover

                order = Order(
                    type=OrderType.TAKE_PROFIT,
                    shares=shares_to_cover,
                    price=take_profit_price,
                    placed_at=current_step,
                    tif=tif,
                )
                self.take_profit_orders.append(order)

                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "take_profit_placed",
                        "shares": shares_to_cover,
                        "price": take_profit_price,
                        "tif": tif.value,
                    }
                )

    # === Private Helper Methods ===
    def _get_reserved_shares(self) -> int:
        """
        Get the total number of shares reserved for open orders.

        Returns:
            int: The total number of shares reserved.
        """
        reserved_sl = sum(order.shares for order in self.stop_loss_orders)
        reserved_tp = sum(order.shares for order in self.take_profit_orders)
        reserved_limit_sell = sum(order.shares for order in self.pending_orders if order.type == OrderType.LIMIT_SELL)
        return reserved_sl + reserved_tp + reserved_limit_sell

    def _process_pending_orders(
        self,
        current_step: int,
        current_price: float,
        current_high: float,
        current_low: float,
        current_open: float,
    ) -> None:
        """Process pending limit orders."""
        remaining_orders: List[Order] = []
        executed_order_details = []

        for order in self.pending_orders:
            executed = False

            # Check for expiration
            expired = False
            if order.tif == OrderTIF.TTL:
                expired = current_step - order.placed_at > self.order_expiration_steps

            if expired:
                if order.type == OrderType.LIMIT_BUY:
                    self.balance += order.cost_reserved
                elif order.type == OrderType.LIMIT_SELL:
                    self.units_held += order.shares

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": f"{order.type.value}_expired",
                        "shares": order.shares,
                        "price": order.price,
                        "reason": "Expired",
                    }
                )
                executed = True

            # === Limit Buy Execution ===
            # Execute if Low price dipped below Limit Price
            elif order.type == OrderType.LIMIT_BUY and current_low <= order.price:
                # Determine execution price (Gap Handling)
                # If Open < Limit, we assume we filled at Open (better price).
                # Otherwise we filled at Limit.
                execution_price = order.price
                if current_open < order.price:
                    execution_price = current_open

                # Refund the cost difference if we got a better price
                actual_cost = order.shares * execution_price * (1 + self.transaction_cost_pct)
                cost_diff = order.cost_reserved - actual_cost
                if cost_diff > 0:
                    self.balance += cost_diff

                # Note: We technically might have reserved too little if execution_price > reserved_price
                # but Limit Buy ensures price <= limit, so cost is always <= reserved.

                self.units_held += order.shares
                executed = True

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "limit_buy_executed",
                        "shares": order.shares,
                        "price": execution_price,
                        "cost": actual_cost,
                    }
                )

            # === Limit Sell Execution ===
            # Execute if High price reached Limit Price
            elif order.type == OrderType.LIMIT_SELL and current_high >= order.price:
                # Determine execution price (Gap Handling)
                # If Open > Limit, we filled at Open (better price).
                execution_price = order.price
                if current_open > order.price:
                    execution_price = current_open

                revenue = order.shares * execution_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                executed = True

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "limit_sell_executed",
                        "shares": order.shares,
                        "price": execution_price,
                        "revenue": revenue,
                    }
                )

            if not executed:
                remaining_orders.append(order)

        # Update the list of pending orders and log any events
        self.pending_orders = remaining_orders
        if executed_order_details:
            self.executed_orders_history.extend(executed_order_details)

    def _process_risk_management_orders(
        self,
        current_step: int,
        current_price: float,
        current_high: float,
        current_low: float,
        current_open: float,
    ) -> None:
        """Process stop-loss and take-profit orders."""
        executed_order_details = []

        # === Process Stop Loss Orders ===
        remaining_stop_loss: List[Order] = []
        for order in self.stop_loss_orders:
            # Check Expiration for TTL
            expired = False
            if order.tif == OrderTIF.TTL:
                expired = current_step - order.placed_at > self.order_expiration_steps

            if expired:
                self.units_held += order.shares
                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "stop_loss_expired",
                        "shares": order.shares,
                        "price": order.price,
                    }
                )
                continue

            # Check Trigger: Low <= Stop Price
            if current_low <= order.price:
                # Determine execution price (Gap Handling)
                # If Open < Stop Price (gap down), we fill at Open (worse price).
                # Otherwise we fill at Stop Price.
                trigger_price = order.price
                fill_price = trigger_price
                if current_open < trigger_price:
                    fill_price = current_open

                # Apply slippage to the fill price
                adjusted_price = fill_price * (1 - self.slippage)
                revenue = order.shares * adjusted_price * (1 - self.transaction_cost_pct)
                self.balance += revenue

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "stop_loss_executed",
                        "shares": order.shares,
                        "trigger_price": trigger_price,
                        "execution_price": adjusted_price,
                        "revenue": revenue,
                    }
                )
            else:
                remaining_stop_loss.append(order)
        self.stop_loss_orders = remaining_stop_loss

        # === Process Take Profit Orders ===
        remaining_take_profit: List[Order] = []
        for order in self.take_profit_orders:
            # Check Expiration
            expired = False
            if order.tif == OrderTIF.TTL:
                expired = current_step - order.placed_at > self.order_expiration_steps

            if expired:
                self.units_held += order.shares
                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "take_profit_expired",
                        "shares": order.shares,
                        "price": order.price,
                    }
                )
                continue

            # Check Trigger: High >= Take Profit Price
            if current_high >= order.price:
                # Determine execution price (Gap Handling)
                # If Open > TP Price (gap up), we fill at Open (better price).
                trigger_price = order.price
                fill_price = trigger_price
                if current_open > trigger_price:
                    fill_price = current_open

                # Apply slippage
                adjusted_price = fill_price * (1 - self.slippage)
                revenue = order.shares * adjusted_price * (1 - self.transaction_cost_pct)
                self.balance += revenue

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "take_profit_executed",
                        "shares": order.shares,
                        "trigger_price": trigger_price,
                        "execution_price": adjusted_price,
                        "revenue": revenue,
                    }
                )
            else:
                remaining_take_profit.append(order)
        self.take_profit_orders = remaining_take_profit

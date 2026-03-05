import pytest

from quantrl_lab.environments.core.types import Actions
from quantrl_lab.environments.stock.components.portfolio import StockPortfolio


@pytest.fixture
def portfolio():
    return StockPortfolio(
        initial_balance=10000.0, transaction_cost_pct=0.001, slippage=0.0001, order_expiration_steps=5
    )


class TestLimitOrders:
    def test_limit_buy_execution(self, portfolio):
        # Place limit buy at $100
        portfolio.place_limit_order(
            action_type=Actions.LimitBuy,
            current_price=105.0,  # Market price
            amount_pct=0.5,
            price_modifier=100.0 / 105.0,  # Result in limit price ~100.0
            current_step=0,
        )
        assert len(portfolio.pending_orders) == 1
        assert portfolio.pending_orders[0].price == pytest.approx(100.0)

        # Step 1: Price 101 (Above limit) - Should NOT execute
        # Limit Buy executes if Low <= Limit
        # current_low defaults to current_price if not provided
        portfolio.process_open_orders(current_step=1, current_price=101.0)
        assert len(portfolio.pending_orders) == 1
        assert portfolio.shares_held == 0

        # Step 2: Price 99 (Below limit) - Should EXECUTE
        portfolio.process_open_orders(current_step=2, current_price=99.0)
        assert len(portfolio.pending_orders) == 0
        assert portfolio.shares_held > 0

    def test_limit_sell_execution(self, portfolio):
        # Setup: Buy some shares first
        portfolio.execute_market_order(Actions.Buy, 100.0, 0.5, 0)
        initial_shares = portfolio.shares_held

        # Place limit sell at $110
        portfolio.place_limit_order(
            action_type=Actions.LimitSell,
            current_price=100.0,
            amount_pct=1.0,  # Sell all
            price_modifier=1.1,  # 100 * 1.1 = 110
            current_step=1,
        )
        # Note: Shares are reserved immediately
        assert portfolio.shares_held == 0
        assert portfolio.total_shares == initial_shares

        # Step 1: Price 105 (Below limit) - Should NOT execute
        # Limit Sell executes if High >= Limit
        portfolio.process_open_orders(current_step=2, current_price=105.0)
        assert len(portfolio.pending_orders) == 1

        # Step 2: Price 115 (Above limit) - Should EXECUTE
        portfolio.process_open_orders(current_step=3, current_price=115.0)
        assert len(portfolio.pending_orders) == 0
        assert portfolio.total_shares == 0
        assert portfolio.balance > 10000.0

    def test_order_expiration(self, portfolio):
        # Place limit buy way below market
        portfolio.place_limit_order(
            action_type=Actions.LimitBuy,
            current_price=100.0,
            amount_pct=0.5,
            price_modifier=0.5,  # Limit = 50
            current_step=10,
        )

        # Advance time 4 steps (10 -> 14) - Should still be there (expiration is 5)
        # created at 10. 10+5 = 15. So at 15 it expires.
        # current_step passed to process_open_orders is "current time"

        portfolio.process_open_orders(current_step=14, current_price=100.0)
        assert len(portfolio.pending_orders) == 1

        # Advance to step 16 - Should expire
        portfolio.process_open_orders(current_step=16, current_price=100.0)
        assert len(portfolio.pending_orders) == 0


class TestCostsAndSlippage:
    def test_transaction_cost_deduction(self, portfolio):
        # Cost is 0.1% (0.001)
        price = 100.0
        amount_pct = 0.5  # $5000 budget

        # Execute buy
        portfolio.execute_market_order(Actions.Buy, price, amount_pct, 0)

        # Check balance reduced
        current_val = portfolio.balance + (portfolio.shares_held * price)
        assert current_val < 10000.0

        # Verify executed order has cost details
        assert len(portfolio.executed_orders_history) == 1
        order_event = portfolio.executed_orders_history[0]
        assert "cost" in order_event

        # Calculate expected cost
        # Price with slippage (0.0001) -> 100.01
        # Cost (0.001) -> 100.01 * 0.001 = 0.10001 per share
        # Total deduction should match logic

"""Tests for deployment/trading/alpaca_trader.py (mocked Alpaca
client)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_trading_client():
    return MagicMock()


@pytest.fixture
def trader(mock_trading_client):
    with patch("quantrl_lab.deployment.trading.alpaca_trader.TradingClient", return_value=mock_trading_client):
        from quantrl_lab.deployment.trading.alpaca_trader import AlpacaTradingClient

        client = AlpacaTradingClient(api_key="test_key", secret_key="test_secret", paper_trading=True)
        client.connect_client()
        return client, mock_trading_client


class TestAlpacaTradingClientInit:
    def test_api_key_set_from_constructor(self):
        from quantrl_lab.deployment.trading.alpaca_trader import AlpacaTradingClient

        client = AlpacaTradingClient(api_key="my_key", secret_key="my_secret")
        assert client.api_key == "my_key"
        assert client.secret_key == "my_secret"

    def test_paper_trading_default_true(self):
        from quantrl_lab.deployment.trading.alpaca_trader import AlpacaTradingClient

        client = AlpacaTradingClient(api_key="k", secret_key="s")
        assert client.paper_trading is True

    def test_paper_trading_false(self):
        from quantrl_lab.deployment.trading.alpaca_trader import AlpacaTradingClient

        client = AlpacaTradingClient(api_key="k", secret_key="s", paper_trading=False)
        assert client.paper_trading is False


class TestConnectClient:
    def test_connect_initializes_trading_client(self):
        mock_tc = MagicMock()
        with patch("quantrl_lab.deployment.trading.alpaca_trader.TradingClient", return_value=mock_tc) as mock_cls:
            from quantrl_lab.deployment.trading.alpaca_trader import AlpacaTradingClient

            client = AlpacaTradingClient(api_key="k", secret_key="s", paper_trading=True)
            client.connect_client()
            mock_cls.assert_called_once_with("k", "s", paper=True)

    def test_connect_paper_flag_passed(self):
        mock_tc = MagicMock()
        with patch("quantrl_lab.deployment.trading.alpaca_trader.TradingClient", return_value=mock_tc) as mock_cls:
            from quantrl_lab.deployment.trading.alpaca_trader import AlpacaTradingClient

            client = AlpacaTradingClient(api_key="k", secret_key="s", paper_trading=False)
            client.connect_client()
            mock_cls.assert_called_once_with("k", "s", paper=False)


class TestGetAccountDetails:
    def test_returns_account(self, trader):
        client, mock_tc = trader
        mock_account = MagicMock()
        mock_tc.get_account.return_value = mock_account
        result = client.get_account_details()
        assert result is mock_account
        mock_tc.get_account.assert_called_once()


class TestCreateOrder:
    def test_create_market_order_calls_submit(self, trader):
        from alpaca.trading.enums import OrderSide

        client, mock_tc = trader
        mock_order = MagicMock()
        mock_tc.submit_order.return_value = mock_order

        result = client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="market")
        assert mock_tc.submit_order.called
        assert result is mock_order

    def test_create_limit_order_requires_limit_price(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError, match="limit_price"):
            client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="limit")

    def test_create_limit_order_with_price(self, trader):
        from alpaca.trading.enums import OrderSide

        client, mock_tc = trader
        mock_tc.submit_order.return_value = MagicMock()
        client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="limit", limit_price=150.0)
        assert mock_tc.submit_order.called

    def test_invalid_order_type_raises(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError, match="Unsupported order_type"):
            client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="super_special")

    def test_neither_qty_nor_notional_raises(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError):
            client.create_order("AAPL", OrderSide.BUY, order_type="market")

    def test_both_qty_and_notional_raises(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError):
            client.create_order("AAPL", OrderSide.BUY, qty=10, notional=1000.0, order_type="market")

    def test_stop_order_requires_stop_price(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError, match="stop_price"):
            client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="stop")

    def test_trailing_stop_requires_trail_percent(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError, match="trail_percent"):
            client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="trailing_stop")

    def test_bracket_order_requires_tp_and_sl(self, trader):
        from alpaca.trading.enums import OrderSide

        client, _ = trader
        with pytest.raises(ValueError, match="take_profit_price"):
            client.create_order("AAPL", OrderSide.BUY, qty=10, order_type="bracket")


class TestGetAllAssets:
    def test_returns_asset_list(self, trader):
        client, mock_tc = trader
        mock_assets = [MagicMock(), MagicMock()]
        mock_tc.get_all_assets.return_value = mock_assets

        result = client.get_all_assets()
        assert result is mock_assets
        mock_tc.get_all_assets.assert_called_once()


class TestGetAllOrders:
    def test_string_symbol_converted_to_list(self, trader):
        client, mock_tc = trader
        mock_tc.get_orders.return_value = []
        client.get_all_orders("AAPL")
        assert mock_tc.get_orders.called

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from loguru import logger

from quantrl_lab.data.interface import (
    DataSource,
    HistoricalDataCapable,
)


class FMPDataSource(
    DataSource,
    HistoricalDataCapable,
):
    """
    Financial Modeling Prep data source for historical stock data.

    Supports both end-of-day (daily) and intraday data.
    Intraday timeframes: 5min, 15min, 30min, 1hour, 4hour
    Daily timeframe: 1d
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    RATE_LIMIT_SLEEP = 1  # seconds
    INTRADAY_TIMEFRAMES = {"5min", "15min", "30min", "1hour", "4hour"}

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP data source.

        Args:
            api_key (str, optional): FMP API key. If not provided, will try to read from
                FMP_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key must be provided or set in FMP_API_KEY environment variable")

    @property
    def source_name(self) -> str:
        return "FinancialModelingPrep"

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """
        Make an HTTP request to the FMP API.

        Args:
            endpoint (str): API endpoint path
            params (Dict[str, Any]): Query parameters

        Returns:
            Any: JSON response data

        Raises:
            requests.HTTPError: If the request fails
        """
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params)

        if response.status_code != 200:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            response.raise_for_status()

        time.sleep(self.RATE_LIMIT_SLEEP)  # Rate limiting
        return response.json()

    def _get_intraday_data(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]],
        timeframe: str,
        nonadjusted: bool = False,
    ) -> pd.DataFrame:
        """
        Get intraday OHLCV data from FMP historical-chart endpoint.

        Args:
            symbol: Stock symbol to fetch data for
            start: Start date for historical data
            end: End date for historical data
            timeframe: Intraday timeframe (5min, 15min, 30min, 1hour, 4hour)
            nonadjusted: If true, returns unadjusted prices (default: False)

        Returns:
            pd.DataFrame: Intraday OHLCV data with standardized column names
        """
        # Convert dates to string format YYYY-MM-DD
        if isinstance(start, datetime):
            start_str = start.strftime("%Y-%m-%d")
        else:
            start_str = pd.to_datetime(start).strftime("%Y-%m-%d")

        if end is not None:
            if isinstance(end, datetime):
                end_str = end.strftime("%Y-%m-%d")
            else:
                end_str = pd.to_datetime(end).strftime("%Y-%m-%d")
        else:
            end_str = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "Fetching {timeframe} intraday data for {symbol} from {start} to {end}",
            timeframe=timeframe,
            symbol=symbol,
            start=start_str,
            end=end_str,
        )

        # Build endpoint: historical-chart/{timeframe}
        endpoint = f"historical-chart/{timeframe}"
        params = {
            "symbol": symbol,
            "from": start_str,
            "to": end_str,
            "nonadjusted": str(nonadjusted).lower(),
        }

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No intraday data found for symbol: {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(f"Empty dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Standardize column names
        df.rename(
            columns={
                'date': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            },
            inplace=True,
        )

        # Add Symbol column
        df['Symbol'] = symbol

        # Add Date column (date only, no time)
        df['Date'] = df['Timestamp'].dt.date

        # Sort by timestamp (FMP may return newest first)
        df.sort_values('Timestamp', inplace=True)

        logger.success(
            "Fetched {n} {timeframe} intraday rows for {symbol}",
            n=len(df),
            timeframe=timeframe,
            symbol=symbol,
        )

        return df

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from FMP (daily or intraday).

        Args:
            symbols: Stock symbol(s) to fetch data for
            start: Start date for historical data
            end: End date for historical data
            timeframe: Timeframe - "1d" for daily, or intraday: "5min", "15min", "30min", "1hour", "4hour"
            **kwargs: Additional arguments including 'nonadjusted' (bool) for intraday data

        Returns:
            pd.DataFrame: OHLCV data with standardized column names

        Raises:
            ValueError: If timeframe is not supported
        """
        # FMP only supports single symbols
        if isinstance(symbols, list):
            if len(symbols) > 1:
                logger.warning("FMP data source only supports single symbol requests. Using first symbol.")
            symbol = symbols[0]
        else:
            symbol = symbols

        # Check if intraday timeframe
        if timeframe in self.INTRADAY_TIMEFRAMES:
            nonadjusted = kwargs.get("nonadjusted", False)
            return self._get_intraday_data(symbol, start, end, timeframe, nonadjusted)

        # Otherwise use daily EOD endpoint
        if timeframe != "1d":
            logger.warning(f"Timeframe {timeframe} not supported by FMP. Using daily (1d) data.")

        # Convert dates to string format YYYY-MM-DD
        if isinstance(start, datetime):
            start_str = start.strftime("%Y-%m-%d")
        else:
            start_str = pd.to_datetime(start).strftime("%Y-%m-%d")

        if end is not None:
            if isinstance(end, datetime):
                end_str = end.strftime("%Y-%m-%d")
            else:
                end_str = pd.to_datetime(end).strftime("%Y-%m-%d")
        else:
            end_str = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "Fetching EOD data for {symbol} from {start} to {end}",
            symbol=symbol,
            start=start_str,
            end=end_str,
        )

        # Build endpoint with query parameters: symbol, from, to
        endpoint = "historical-price-eod/full"
        params = {
            "symbol": symbol,
            "from": start_str,
            "to": end_str,
        }

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical data found for symbol: {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(f"Empty dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])

        # Standardize column names to match other data sources
        df.rename(
            columns={
                'date': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            },
            inplace=True,
        )

        # Add Symbol column
        df['Symbol'] = symbol

        # Add Date column (date only, no time)
        df['Date'] = df['Timestamp'].dt.date

        # Sort by timestamp (FMP returns newest first)
        df.sort_values('Timestamp', inplace=True)

        logger.success(
            "Fetched {n} OHLCV rows for {symbol}",
            n=len(df),
            symbol=symbol,
        )

        return df

    def get_historical_grades(self, symbol: str) -> pd.DataFrame:
        """
        Get historical analyst grades for a symbol.

        Args:
            symbol: Stock symbol to fetch data for

        Returns:
            pd.DataFrame: Historical grades data
        """
        endpoint = "grades-historical"
        params = {"symbol": symbol}

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical grades found for symbol: {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(f"Empty grades dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

        logger.success(
            "Fetched {n} historical grades for {symbol}",
            n=len(df),
            symbol=symbol,
        )

        return df

    def get_historical_rating(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical ratings for a symbol.

        Args:
            symbol: Stock symbol to fetch data for
            limit: Number of records to return (default: 100)

        Returns:
            pd.DataFrame: Historical ratings data
        """
        endpoint = "ratings-historical"
        params = {"symbol": symbol, "limit": limit}

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical ratings found for symbol: {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(f"Empty ratings dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

        logger.success(
            "Fetched {n} historical ratings for {symbol}",
            n=len(df),
            symbol=symbol,
        )

        return df

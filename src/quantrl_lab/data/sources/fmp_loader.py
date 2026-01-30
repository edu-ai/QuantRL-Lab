import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from quantrl_lab.data.interface import (
    AnalystDataCapable,
    DataSource,
    HistoricalDataCapable,
)
from quantrl_lab.data.utils import (
    HTTPRequestWrapper,
    RetryStrategy,
    convert_to_dataframe_safe,
    format_date_to_string,
    get_single_symbol,
    log_dataframe_info,
    normalize_date_range,
    standardize_ohlcv_dataframe,
)


class FMPDataSource(
    DataSource,
    HistoricalDataCapable,
    AnalystDataCapable,
):
    """
    Financial Modeling Prep data source for historical stock data and
    analyst insights.

    Supports both end-of-day (daily) and intraday data.
    Intraday timeframes: 5min, 15min, 30min, 1hour, 4hour
    Daily timeframe: 1d

    Also provides analyst grades and ratings data.
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

        # Initialize HTTP request wrapper with retry logic
        self._request_wrapper = HTTPRequestWrapper(
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            rate_limit_delay=self.RATE_LIMIT_SLEEP,
            timeout=30.0,
        )

    @property
    def source_name(self) -> str:
        return "FinancialModelingPrep"

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """
        Make an HTTP request to the FMP API with retry logic.

        Args:
            endpoint (str): API endpoint path
            params (Dict[str, Any]): Query parameters

        Returns:
            Any: JSON response data

        Raises:
            requests.HTTPError: If the request fails after retries
        """
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        return self._request_wrapper.make_request(
            url=url,
            method="GET",
            params=params,
            raise_on_error=True,
        )

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
        # Normalize dates using utility
        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        start_str = format_date_to_string(start_dt)
        end_str = format_date_to_string(end_dt)

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

        # Make API request
        data = self._make_request(endpoint, params)

        # Safely convert to DataFrame
        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=symbol)
        if df.empty:
            return df

        # Standardize using utility function
        column_mapping = {
            'date': 'Timestamp',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }

        df = standardize_ohlcv_dataframe(
            df,
            column_mapping=column_mapping,
            symbol=symbol,
            timestamp_col='Timestamp',
            add_date=True,
            sort_data=True,
            convert_numeric=True,
        )

        log_dataframe_info(df, f"Fetched {timeframe} intraday data", symbol=symbol)
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
        # FMP only supports single symbols - extract first symbol
        symbol = get_single_symbol(symbols, warn_on_multiple=True)

        # Check if intraday timeframe
        if timeframe in self.INTRADAY_TIMEFRAMES:
            nonadjusted = kwargs.get("nonadjusted", False)
            return self._get_intraday_data(symbol, start, end, timeframe, nonadjusted)

        # Otherwise use daily EOD endpoint
        if timeframe != "1d":
            logger.warning(f"Timeframe {timeframe} not supported by FMP. Using daily (1d) data.")

        # Normalize dates using utility
        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        start_str = format_date_to_string(start_dt)
        end_str = format_date_to_string(end_dt)

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

        # Make API request
        data = self._make_request(endpoint, params)

        # Safely convert to DataFrame
        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=symbol)
        if df.empty:
            return df

        # Standardize using utility function
        column_mapping = {
            'date': 'Timestamp',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }

        df = standardize_ohlcv_dataframe(
            df,
            column_mapping=column_mapping,
            symbol=symbol,
            timestamp_col='Timestamp',
            add_date=True,
            sort_data=True,
            convert_numeric=True,
        )

        log_dataframe_info(df, "Fetched EOD data", symbol=symbol)
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

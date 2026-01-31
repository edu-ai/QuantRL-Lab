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
    normalize_symbols,
    standardize_ohlcv_dataframe,
    validate_symbols,
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

    Also provides:
    - Analyst grades and ratings data
    - Historical sector performance data
    - Historical industry performance data
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

    def get_historical_sector_performance(self, sector: str) -> pd.DataFrame:
        """
        Get historical performance data for a specific market sector.

        This endpoint provides historical performance metrics for market sectors,
        allowing analysis of sector trends and performance over time.

        Args:
            sector: Market sector name (e.g., "Energy", "Technology", "Healthcare",
                "Financials", "Consumer Cyclical", "Industrials", "Basic Materials",
                "Consumer Defensive", "Real Estate", "Utilities", "Communication Services")

        Returns:
            pd.DataFrame: Historical sector performance data with columns including:
                - date: Performance date
                - sector: Sector name
                - performance metrics (varies by API response)

        Raises:
            ValueError: If sector is invalid or API request fails

        Example:
            >>> source = FMPDataSource()
            >>> df = source.get_historical_sector_performance("Energy")
            >>> print(df.head())
        """
        if not sector or not isinstance(sector, str):
            raise ValueError("Sector must be a non-empty string")

        logger.info("Fetching historical performance for sector: {sector}", sector=sector)

        endpoint = "historical-sector-performance"
        params = {"sector": sector}

        # Make API request
        data = self._make_request(endpoint, params)

        # Validate response
        if not data or not isinstance(data, list):
            logger.warning(f"No historical sector performance data found for sector: {sector}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=sector)

        if df.empty:
            logger.warning(f"Empty sector performance dataset returned for sector: {sector}")
            return pd.DataFrame()

        # Convert date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

        log_dataframe_info(df, "Fetched sector performance", symbol=sector)

        logger.success(
            "Fetched {n} records of historical sector performance for {sector}",
            n=len(df),
            sector=sector,
        )

        return df

    def get_historical_industry_performance(self, industry: str) -> pd.DataFrame:
        """
        Get historical performance data for a specific industry.

        This endpoint provides historical performance metrics for industries,
        enabling long-term trend analysis and industry evolution tracking.

        Args:
            industry: Industry name (e.g., "Biotechnology", "Software", "Banks",
                "Oil & Gas", "Semiconductors", "Insurance", "Auto Manufacturers",
                "Pharmaceuticals", "Consumer Electronics", "Aerospace & Defense")

        Returns:
            pd.DataFrame: Historical industry performance data with columns including:
                - date: Performance date
                - industry: Industry name
                - performance metrics (varies by API response)

        Raises:
            ValueError: If industry is invalid or API request fails

        Example:
            >>> source = FMPDataSource()
            >>> df = source.get_historical_industry_performance("Biotechnology")
            >>> print(df.head())
        """
        if not industry or not isinstance(industry, str):
            raise ValueError("Industry must be a non-empty string")

        logger.info("Fetching historical performance for industry: {industry}", industry=industry)

        endpoint = "historical-industry-performance"
        params = {"industry": industry}

        # Make API request
        data = self._make_request(endpoint, params)

        # Validate response
        if not data or not isinstance(data, list):
            logger.warning(f"No historical industry performance data found for industry: {industry}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=industry)

        if df.empty:
            logger.warning(f"Empty industry performance dataset returned for industry: {industry}")
            return pd.DataFrame()

        # Convert date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

        log_dataframe_info(df, "Fetched industry performance", symbol=industry)

        logger.success(
            "Fetched {n} records of historical industry performance for {industry}",
            n=len(df),
            industry=industry,
        )

        return df

    def get_company_profile(self, symbol: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get company profile information including sector, industry, and
        key metrics.

        This endpoint provides comprehensive company information including business
        description, sector/industry classification, executive information, and
        key financial metrics.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT") or list of symbols
                (only first symbol will be used if list is provided)

        Returns:
            pd.DataFrame: Company profile data with columns including:
                - symbol: Stock ticker
                - companyName: Full company name
                - sector: Sector classification (e.g., "Technology")
                - industry: Industry classification (e.g., "Consumer Electronics")
                - description: Business description
                - ceo: Chief Executive Officer name
                - website: Company website URL
                - exchange: Stock exchange
                - exchangeShortName: Exchange abbreviation
                - mktCap: Market capitalization
                - price: Current stock price
                - beta: Stock beta
                - volAvg: Average volume
                - currency: Trading currency
                - ipoDate: Initial public offering date
                - address, city, state, zip, country: Headquarters location
                - phone: Contact phone number
                - fullTimeEmployees: Number of employees
                - image: Company logo URL
                - isEtf, isActivelyTrading, isFund, isAdr: Asset type flags

        Raises:
            ValueError: If symbol is invalid or API request fails

        Example:
            >>> source = FMPDataSource()
            >>> profile = source.get_company_profile("AAPL")
            >>> print(f"Sector: {profile.iloc[0]['sector']}")
            >>> print(f"Industry: {profile.iloc[0]['industry']}")
            >>> print(f"CEO: {profile.iloc[0]['ceo']}")

        Use Cases:
            - Get sector/industry classification for stocks
            - Screen stocks by sector or industry
            - Retrieve company metadata for analysis
            - Build company information datasets
        """
        # Normalize and validate symbol
        symbols = normalize_symbols(symbol)
        validate_symbols(symbols)
        symbol = get_single_symbol(symbols)

        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        logger.info("Fetching company profile for: {symbol}", symbol=symbol)

        endpoint = "profile"
        params = {"symbol": symbol}

        # Make API request
        data = self._make_request(endpoint, params)

        # Validate response
        if not data or not isinstance(data, list):
            logger.warning(f"No company profile data found for symbol: {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=symbol)

        if df.empty:
            logger.warning(f"Empty company profile dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        # Log the company info
        if not df.empty:
            company_name = df.iloc[0].get('companyName', 'Unknown')
            sector = df.iloc[0].get('sector', 'N/A')
            industry = df.iloc[0].get('industry', 'N/A')

            logger.success(
                "Fetched company profile for {symbol}: {name} ({sector} - {industry})",
                symbol=symbol,
                name=company_name,
                sector=sector,
                industry=industry,
            )

        return df

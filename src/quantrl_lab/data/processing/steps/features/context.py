"""Market context processing steps."""

import pandas as pd
from rich.console import Console

from quantrl_lab.data.processing.processor import ProcessingMetadata

console = Console()


class MarketContextStep:
    """
    Merge broad market context (Sector/Industry performance) into the
    DataFrame.

    This step allows the agent to see how the specific stock's sector or
    industry is performing relative to the stock itself.

    Attributes:
        sector_perf_df (pd.DataFrame): Historical sector performance.
        industry_perf_df (pd.DataFrame): Historical industry performance.
    """

    def __init__(self, sector_perf_df: pd.DataFrame = None, industry_perf_df: pd.DataFrame = None):
        self.sector_perf_df = sector_perf_df
        self.industry_perf_df = industry_perf_df

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Merge sector and industry data.

        Args:
            data: Input OHLCV DataFrame
            metadata: Processing metadata

        Returns:
            DataFrame with added context features (prefixed with sector_ or industry_)
        """
        if (self.sector_perf_df is None or self.sector_perf_df.empty) and (
            self.industry_perf_df is None or self.industry_perf_df.empty
        ):
            return data

        df = data.copy()

        # Setup index for merging
        date_col = None
        temp_index = False
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find date column
            for col in ['Timestamp', 'Date', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    date_col = col
                    temp_index = True
                    break

        # Normalize index to tz-naive UTC midnight to prevent join errors
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            df.index = df.index.normalize()

        # --- Merge Sector Performance ---
        if self.sector_perf_df is not None and not self.sector_perf_df.empty:
            sector_df = self.sector_perf_df.copy()
            if 'date' in sector_df.columns:
                sector_df['date'] = pd.to_datetime(sector_df['date'])
                sector_df.set_index('date', inplace=True)

                if isinstance(sector_df.index, pd.DatetimeIndex):
                    if sector_df.index.tz is not None:
                        sector_df.index = sector_df.index.tz_convert('UTC').tz_localize(None)
                    sector_df.index = sector_df.index.normalize()

                # Keep numeric columns only for performance metrics
                numeric_cols = sector_df.select_dtypes(include=['number']).columns
                sector_df = sector_df[numeric_cols]

                # Add prefix
                sector_df = sector_df.add_prefix('sector_')

                # Join
                df = df.join(sector_df, how='left')

        # --- Merge Industry Performance ---
        if self.industry_perf_df is not None and not self.industry_perf_df.empty:
            ind_df = self.industry_perf_df.copy()
            if 'date' in ind_df.columns:
                ind_df['date'] = pd.to_datetime(ind_df['date'])
                ind_df.set_index('date', inplace=True)

                if isinstance(ind_df.index, pd.DatetimeIndex):
                    if ind_df.index.tz is not None:
                        ind_df.index = ind_df.index.tz_convert('UTC').tz_localize(None)
                    ind_df.index = ind_df.index.normalize()

                numeric_cols = ind_df.select_dtypes(include=['number']).columns
                ind_df = ind_df[numeric_cols]

                ind_df = ind_df.add_prefix('industry_')

                df = df.join(ind_df, how='left')

        # Restore index if we changed it temporarily
        if temp_index and date_col:
            df.reset_index(inplace=True)

        return df

    def get_step_name(self) -> str:
        return "Market Context Enrichment"

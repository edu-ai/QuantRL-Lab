"""Analyst data processing steps."""

import pandas as pd
from rich.console import Console

from quantrl_lab.data.processing.processor import ProcessingMetadata

console = Console()


class AnalystEstimatesStep:
    """
    Merge analyst grades and ratings into the DataFrame.

    This step merges historical analyst data (grades, ratings) onto the main
    OHLCV DataFrame based on timestamps. Since analyst updates are sparse,
    values are forward-filled to represent the "current" analyst consensus
    at each time step.

    Attributes:
        grades_df (pd.DataFrame): Historical grades data.
        ratings_df (pd.DataFrame): Historical ratings data.
    """

    def __init__(self, grades_df: pd.DataFrame = None, ratings_df: pd.DataFrame = None):
        self.grades_df = grades_df
        self.ratings_df = ratings_df

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Merge and forward-fill analyst data.

        Args:
            data: Input OHLCV DataFrame (must have datetime index or 'Date'/'Timestamp' column)
            metadata: Processing metadata

        Returns:
            DataFrame with added analyst features
        """
        if (self.grades_df is None or self.grades_df.empty) and (self.ratings_df is None or self.ratings_df.empty):
            return data

        # Ensure working copy and set index to datetime for merging
        df = data.copy()

        # Identify date column for merging
        date_col = None
        if isinstance(df.index, pd.DatetimeIndex):
            pass  # Index is already good
        elif 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            date_col = 'Timestamp'
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            date_col = 'Date'

        # Normalize index to tz-naive UTC and midnight to prevent join errors
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            df.index = df.index.normalize()

        # Helper to prepare month key for joining
        # We use to_period('M') to match "2023-01-03" (Data) with "2023-01-01" (Grade)
        df['_join_month'] = df.index.to_period('M')

        # --- Process Grades ---
        if self.grades_df is not None and not self.grades_df.empty:
            grades = self.grades_df.copy()
            if 'date' in grades.columns:
                grades['date'] = pd.to_datetime(grades['date'])

                # Create join key
                grades['_join_month'] = grades['date'].dt.to_period('M')

                # Deduplicate: Keep the last rating for the month if multiple exist
                # This prevents row explosion (Cartesian product) if FMP has >1 record/month
                grades = grades.sort_values('date').drop_duplicates(subset=['_join_month'], keep='last')

                # Drop redundant columns
                grades = grades.drop(columns=['symbol', 'date'], errors='ignore')

                # Merge on month key
                # We use reset_index() on df to preserve the DatetimeIndex during merge
                # then set it back.
                df_reset = df.reset_index()

                # Merge
                merged = pd.merge(df_reset, grades, on='_join_month', how='left', suffixes=('', '_grade'))

                # Restore index
                if date_col:
                    merged.set_index(date_col, inplace=True)
                else:
                    # Fallback if date_col wasn't explicitly tracked (shouldn't happen given logic above)
                    merged.set_index(df.index.name or 'index', inplace=True)

                df = merged

        # --- Process Ratings ---
        if self.ratings_df is not None and not self.ratings_df.empty:
            ratings = self.ratings_df.copy()
            if 'date' in ratings.columns:
                ratings['date'] = pd.to_datetime(ratings['date'])

                # Create join key
                ratings['_join_month'] = ratings['date'].dt.to_period('M')

                # Deduplicate
                ratings = ratings.sort_values('date').drop_duplicates(subset=['_join_month'], keep='last')

                # Drop redundant columns
                ratings = ratings.drop(columns=['symbol', 'rating', 'date'], errors='ignore')

                # Merge
                df_reset = df.reset_index()

                merged = pd.merge(df_reset, ratings, on='_join_month', how='left', suffixes=('', '_rating'))

                # Restore index
                if date_col:
                    merged.set_index(date_col, inplace=True)
                else:
                    merged.set_index(df.index.name or 'index', inplace=True)

                df = merged

        # Cleanup join key
        if '_join_month' in df.columns:
            df = df.drop(columns=['_join_month'])

        # Reset index if we changed it
        if date_col:
            df.reset_index(inplace=True)

        return df

    def get_step_name(self) -> str:
        return "Analyst Estimates Enrichment"

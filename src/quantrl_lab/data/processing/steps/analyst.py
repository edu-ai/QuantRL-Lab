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

        # --- Process Grades ---
        if self.grades_df is not None and not self.grades_df.empty:
            grades = self.grades_df.copy()
            if 'date' in grades.columns:
                grades['date'] = pd.to_datetime(grades['date'])
                grades.set_index('date', inplace=True)

                # Normalize to tz-naive
                if isinstance(grades.index, pd.DatetimeIndex):
                    if grades.index.tz is not None:
                        grades.index = grades.index.tz_convert('UTC').tz_localize(None)
                    grades.index = grades.index.normalize()

                # Drop redundant columns (symbol, date already used as index)
                grades = grades.drop(columns=['symbol'], errors='ignore')

                # Merge with left join
                df = df.join(grades, how='left', rsuffix='_grade')

        # --- Process Ratings ---
        if self.ratings_df is not None and not self.ratings_df.empty:
            ratings = self.ratings_df.copy()
            if 'date' in ratings.columns:
                ratings['date'] = pd.to_datetime(ratings['date'])
                ratings.set_index('date', inplace=True)

                # Normalize to tz-naive
                if isinstance(ratings.index, pd.DatetimeIndex):
                    if ratings.index.tz is not None:
                        ratings.index = ratings.index.tz_convert('UTC').tz_localize(None)
                    ratings.index = ratings.index.normalize()

                # Drop redundant columns (symbol, date already used as index, rating is categorical)
                ratings = ratings.drop(columns=['symbol', 'rating'], errors='ignore')

                # Merge with left join
                df = df.join(ratings, how='left', rsuffix='_rating')

        # Reset index if we changed it
        if date_col:
            df.reset_index(inplace=True)

        return df

    def get_step_name(self) -> str:
        return "Analyst Estimates Enrichment"

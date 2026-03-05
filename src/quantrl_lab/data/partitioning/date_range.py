"""Date range-based data splitter for time series data."""

from typing import Dict, Tuple

import pandas as pd


class DateRangeSplitter:
    """
    Split DataFrame by explicit date ranges.

    This splitter divides data based on specified date ranges,
    useful for creating specific train/test periods.

    Example:
        >>> splitter = DateRangeSplitter({
        ...     "train": ("2020-01-01", "2021-12-31"),
        ...     "test": ("2022-01-01", "2022-12-31")
        ... })
        >>> splits = splitter.split(df)
        >>> train_df = splits["train"]
        >>> test_df = splits["test"]
    """

    def __init__(self, ranges: Dict[str, Tuple[str, str]]):
        """
        Initialize DateRangeSplitter.

        Args:
            ranges (Dict[str, Tuple[str, str]]): Dictionary mapping split names to
                (start_date, end_date) tuples. Dates can be strings or datetime objects.
                Example: {"train": ("2020-01-01", "2021-12-31")}

        Raises:
            ValueError: If ranges are invalid or empty.
        """
        if not ranges:
            raise ValueError("Ranges dictionary cannot be empty")

        # Validate range format
        for name, date_range in ranges.items():
            if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
                raise ValueError(
                    f"Invalid range for '{name}': {date_range}. Must be a tuple/list of (start_date, end_date)"
                )

            start_date, end_date = date_range
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                if start_dt > end_dt:
                    raise ValueError(f"Start date {start_date} is after end date {end_date} for '{name}'")
            except Exception as e:
                raise ValueError(f"Invalid dates for '{name}': {e}")

        self.ranges = ranges

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame by date ranges.

        Args:
            df (pd.DataFrame): Input DataFrame to split. Must have a date column
                (Date, date, timestamp, or Timestamp).

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of split DataFrames.

        Raises:
            ValueError: If DataFrame is empty or date column not found.
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")

        # Find date column
        date_column = next((col for col in ["Date", "date", "timestamp", "Timestamp"] if col in df.columns), None)

        if not date_column:
            raise ValueError(
                "Date column not found. DataFrame must contain one of: 'Date', 'date', 'timestamp', 'Timestamp'"
            )

        # Prepare DataFrame
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None)
        df = df.sort_values(by=date_column).reset_index(drop=True)

        split_data = {}
        metadata_ranges = {}
        metadata_shapes = {}

        for name, (start_date, end_date) in self.ranges.items():
            # Convert to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Filter data
            mask = (df[date_column] >= start_dt) & (df[date_column] <= end_dt)
            subset = df[mask].copy()
            split_data[name] = subset

            # Track metadata
            if not subset.empty:
                metadata_ranges[name] = {
                    "start": subset[date_column].min().strftime("%Y-%m-%d"),
                    "end": subset[date_column].max().strftime("%Y-%m-%d"),
                }
            else:
                metadata_ranges[name] = {
                    "start": start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d"),
                    "end": end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d"),
                }

            metadata_shapes[name] = subset.shape

        # Store metadata for get_metadata() call
        self._last_metadata = {
            "date_ranges": metadata_ranges,
            "final_shapes": metadata_shapes,
        }

        return split_data

    def get_metadata(self) -> Dict:
        """
        Return metadata about the split.

        Returns:
            Dict: Dictionary containing:
                - type: "date_range"
                - ranges: Configuration used
                - date_ranges: Actual date ranges in each split
                - final_shapes: Shape of each split DataFrame
        """
        metadata = {
            "type": "date_range",
            "ranges": {
                name: (
                    start if isinstance(start, str) else start.strftime("%Y-%m-%d"),
                    end if isinstance(end, str) else end.strftime("%Y-%m-%d"),
                )
                for name, (start, end) in self.ranges.items()
            },
        }

        # Add metadata from last split operation if available
        if hasattr(self, "_last_metadata"):
            metadata.update(self._last_metadata)

        return metadata

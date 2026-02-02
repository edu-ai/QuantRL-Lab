"""Base protocol for data splitting strategies."""

from typing import Dict, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSplitter(Protocol):
    """
    Protocol for data splitting strategies.

    This protocol defines the interface for splitting DataFrames into
    multiple subsets (e.g., train/test/validation) using different
    strategies.
    """

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame into multiple subsets.

        Args:
            df (pd.DataFrame): Input DataFrame to split. Must be sorted by date/time.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping split names to DataFrames.
                Example: {"train": train_df, "test": test_df}

        Raises:
            ValueError: If DataFrame is empty or required columns are missing.
        """
        ...

    def get_metadata(self) -> Dict:
        """
        Return metadata about the splitting configuration.

        Returns:
            Dict: Dictionary containing split configuration details.
        """
        ...

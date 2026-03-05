"""Base protocol for feature generators."""

from typing import Dict, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureGenerator(Protocol):
    """
    Protocol for feature generation strategies.

    Generators add features to existing DataFrames without modifying the
    original structure (e.g., adding technical indicators, sentiment
    scores).
    """

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate additional features for DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame to process.
            **kwargs: Additional parameters specific to the generator.

        Returns:
            pd.DataFrame: DataFrame with new feature columns added.

        Raises:
            ValueError: If data is invalid or required columns missing.
        """
        ...

    def get_metadata(self) -> Dict:
        """
        Return metadata about feature generation performed.

        Returns:
            Dict: Dictionary containing feature generation details.
        """
        ...

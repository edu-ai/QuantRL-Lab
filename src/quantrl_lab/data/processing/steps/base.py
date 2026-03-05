"""
Base protocol and types for processing steps in the data pipeline.

This module defines the core abstractions for building composable data
processing pipelines. Each step performs a single transformation and can
be chained together.
"""

from typing import Protocol, runtime_checkable

import pandas as pd

from quantrl_lab.data.processing.processor import ProcessingMetadata


@runtime_checkable
class ProcessingStep(Protocol):
    """
    Protocol for pipeline processing steps.

    Steps are composable transformations that take a DataFrame and metadata,
    apply a transformation, and return the modified DataFrame. Steps should
    update the metadata to track what transformations were applied.

    Example:
        >>> step = TechnicalIndicatorStep(indicators=["SMA", "RSI"])
        >>> result_df = step.process(input_df, metadata)
        >>> assert "SMA_20" in result_df.columns
    """

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Apply transformation to DataFrame.

        Args:
            data: Input DataFrame to process
            metadata: Processing metadata to track transformations

        Returns:
            Transformed DataFrame (may be modified in-place or copied)

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        ...

    def get_step_name(self) -> str:
        """
        Return human-readable name of this processing step.

        Returns:
            Step name (e.g., "Technical Indicators", "Sentiment Enrichment")
        """
        ...

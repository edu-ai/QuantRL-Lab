"""Technical indicator processing step."""

from typing import Dict, List, Optional, Union

import pandas as pd
from rich.console import Console

from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator
from quantrl_lab.data.processing.processor import ProcessingMetadata

console = Console()


class TechnicalIndicatorStep:
    """
    Apply technical indicators to DataFrame.

    This step wraps TechnicalFeatureGenerator to add technical indicators
    as new columns. Indicators can be specified as strings (use defaults)
    or dicts (with custom parameters).

    Example:
        >>> step = TechnicalIndicatorStep(indicators=["SMA", {"RSI": {"window": 14}}])
        >>> result = step.process(df, metadata)
    """

    def __init__(self, indicators: Optional[List[Union[str, Dict]]] = None):
        """
        Initialize technical indicator step.

        Args:
            indicators: List of indicators to apply. Can be strings ("SMA")
                or dicts ({"SMA": {"window": 20}}).
        """
        self.indicators = indicators or []

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Apply technical indicators to DataFrame.

        Args:
            data: Input DataFrame with OHLCV data
            metadata: Processing metadata (updated with applied indicators)

        Returns:
            DataFrame with technical indicator columns added

        Raises:
            ValueError: If required columns are missing
        """
        if not self.indicators:
            return data.copy()

        try:
            generator = TechnicalFeatureGenerator(self.indicators)
            result = generator.generate(data)

            # Update metadata
            metadata.technical_indicators = self.indicators

            return result
        except ValueError as e:
            raise e
        except Exception as e:
            console.print(f"[red]❌ Failed to apply technical indicators: {e}[/red]")
            return data.copy()

    def get_step_name(self) -> str:
        """Return step name."""
        return "Technical Indicators"

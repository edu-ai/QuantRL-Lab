import numpy as np
import pandas as pd

from quantrl_lab.data.config import config
from quantrl_lab.data.processing.processor import ProcessingMetadata


class TimeFeatureStep:
    """
    Generates cyclical time features (sin/cos) from the date column.

    This step should be run BEFORE any column cleanup steps that might
    remove the date column. It creates numeric features that preserve
    time information in a way suitable for machine learning models.
    """

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Add cyclical time features to the DataFrame.

        Args:
            data: Input DataFrame containing a date column
            metadata: Processing metadata to update

        Returns:
            DataFrame with added time features
        """
        # Find the valid date column
        date_col = next((col for col in config.DATE_COLUMNS if col in data.columns), None)

        if date_col:
            # Ensure datetime format
            dates = pd.to_datetime(data[date_col])

            # 1. Day of Week Encoding (0-6)
            # sin(2 * pi * day / 7), cos(2 * pi * day / 7)
            # This captures weekly patterns (e.g., Friday effects)
            data["day_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
            data["day_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)

            # 2. Month Encoding (0-11)
            # sin(2 * pi * (month-1) / 12), cos(2 * pi * (month-1) / 12)
            # This captures seasonal/yearly patterns (e.g., January effect)
            # Note: We subtract 1 from month (1-12) to get 0-11 range
            data["month_sin"] = np.sin(2 * np.pi * (dates.dt.month - 1) / 12)
            data["month_cos"] = np.cos(2 * np.pi * (dates.dt.month - 1) / 12)

            # Add to metadata so we know these were added
            metadata.technical_indicators.append("time_features")

        return data

    def get_step_name(self) -> str:
        """Return step name."""
        return "Time Feature Enrichment"

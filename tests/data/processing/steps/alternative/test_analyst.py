"""Tests for AnalystEstimatesStep."""

import numpy as np
import pandas as pd

from quantrl_lab.data.processing.processor import ProcessingMetadata
from quantrl_lab.data.processing.steps.alternative.analyst import AnalystEstimatesStep


class TestAnalystEstimatesStep:
    """Test AnalystEstimatesStep functionality."""

    def test_monthly_join_alignment(self):
        """
        Test that monthly analyst grades are correctly applied to daily
        data.

        Scenario:
        - Analyst grade released on 2025-01-01 (Holiday/Weekend).
        - Trading data starts on 2025-01-03.
        - Strict join fails (Jan 1 != Jan 3).
        - Monthly join (Jan == Jan) should succeed.
        """
        # 1. Create Daily OHLCV Data (Starts Jan 3rd)
        dates = pd.date_range(start="2025-01-03", end="2025-01-10", freq="D")
        ohlcv_df = pd.DataFrame({"Date": dates, "Close": [100.0] * len(dates), "Volume": [1000] * len(dates)})

        # 2. Create Monthly Analyst Data (Jan 1st)
        grades_df = pd.DataFrame(
            {"date": [pd.Timestamp("2025-01-01")], "analystRatingsStrongBuy": [10.0], "symbol": ["TEST"]}
        )

        step = AnalystEstimatesStep(grades_df=grades_df)
        metadata = ProcessingMetadata()

        # 3. Process
        result = step.process(ohlcv_df, metadata)

        # 4. Verify
        # Jan 3rd should have the grade from Jan 1st (same month)
        val_jan3 = result.loc[result["Date"] == "2025-01-03", "analystRatingsStrongBuy"].iloc[0]

        assert not np.isnan(val_jan3), "Analyst data should not be NaN for Jan 3rd"
        assert val_jan3 == 10.0

    def test_monthly_join_updates_correctly(self):
        """Test that grades update when the month changes."""
        # Data spans Jan and Feb
        dates = pd.date_range(start="2025-01-28", end="2025-02-03", freq="D")
        ohlcv_df = pd.DataFrame({"Date": dates, "Close": [100.0] * len(dates)})

        # Grades for Jan 1 and Feb 1
        grades_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")],
                "analystRatingsStrongBuy": [10.0, 20.0],
                "symbol": ["TEST", "TEST"],
            }
        )

        step = AnalystEstimatesStep(grades_df=grades_df)
        result = step.process(ohlcv_df, ProcessingMetadata())

        # Jan 31 should have Jan grade (10.0)
        val_jan31 = result.loc[result["Date"] == "2025-01-31", "analystRatingsStrongBuy"].iloc[0]
        assert val_jan31 == 10.0

        # Feb 1 should have Feb grade (20.0)
        val_feb1 = result.loc[result["Date"] == "2025-02-01", "analystRatingsStrongBuy"].iloc[0]
        assert val_feb1 == 20.0

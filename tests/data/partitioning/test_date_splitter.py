"""Tests for DateRangeSplitter."""

import pandas as pd
import pytest

from quantrl_lab.data.partitioning import DateRangeSplitter


class TestDateRangeSplitterInit:
    """Test DateRangeSplitter initialization."""

    def test_valid_ranges(self):
        """Test initialization with valid date ranges."""
        ranges = {"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")}
        splitter = DateRangeSplitter(ranges)
        assert splitter.ranges == ranges

    def test_single_range(self):
        """Test initialization with single date range."""
        ranges = {"train": ("2020-01-01", "2021-12-31")}
        splitter = DateRangeSplitter(ranges)
        assert splitter.ranges == ranges

    def test_multiple_ranges(self):
        """Test initialization with multiple date ranges."""
        ranges = {
            "train": ("2020-01-01", "2020-12-31"),
            "val": ("2021-01-01", "2021-12-31"),
            "test": ("2022-01-01", "2022-12-31"),
        }
        splitter = DateRangeSplitter(ranges)
        assert len(splitter.ranges) == 3

    def test_empty_ranges_raises_error(self):
        """Test that empty ranges dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DateRangeSplitter({})

    def test_invalid_range_format_raises_error(self):
        """Test that invalid range format raises ValueError."""
        # Single value instead of tuple
        with pytest.raises(ValueError, match="Invalid range"):
            DateRangeSplitter({"train": "2020-01-01"})

    def test_wrong_number_of_dates_raises_error(self):
        """Test that wrong number of dates raises ValueError."""
        # Three dates instead of two
        with pytest.raises(ValueError, match="Invalid range"):
            DateRangeSplitter({"train": ("2020-01-01", "2021-01-01", "2022-01-01")})

    def test_start_after_end_raises_error(self):
        """Test that start date after end date raises ValueError."""
        with pytest.raises(ValueError, match="after end date"):
            DateRangeSplitter({"train": ("2022-01-01", "2020-01-01")})

    def test_invalid_date_string_raises_error(self):
        """Test that invalid date string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dates"):
            DateRangeSplitter({"train": ("not-a-date", "2020-01-01")})

    def test_datetime_objects_as_ranges(self):
        """Test initialization with datetime objects."""
        from datetime import datetime

        ranges = {
            "train": (datetime(2020, 1, 1), datetime(2021, 12, 31)),
            "test": (datetime(2022, 1, 1), datetime(2022, 12, 31)),
        }
        splitter = DateRangeSplitter(ranges)
        assert splitter.ranges == ranges


class TestDateRangeSplitterSplit:
    """Test DateRangeSplitter.split() method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data spanning multiple years."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": range(len(dates)),
                "High": range(100, 100 + len(dates)),
                "Low": range(0, len(dates)),
                "Close": range(50, 50 + len(dates)),
                "Volume": range(1000, 1000 + len(dates)),
            }
        )

    def test_split_two_ranges(self, sample_data):
        """Test splitting data into two date ranges."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splits = splitter.split(sample_data)

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) == 731  # 2020-2021 (2 years, 1 leap year)
        assert len(splits["test"]) == 365  # 2022

    def test_split_three_ranges(self, sample_data):
        """Test splitting data into three date ranges."""
        splitter = DateRangeSplitter(
            {
                "train": ("2020-01-01", "2020-12-31"),
                "val": ("2021-01-01", "2021-12-31"),
                "test": ("2022-01-01", "2022-12-31"),
            }
        )
        splits = splitter.split(sample_data)

        assert len(splits) == 3
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_split_maintains_date_boundaries(self, sample_data):
        """Test that splits respect date boundaries."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splits = splitter.split(sample_data)

        # Train should end in 2021
        train_last_date = splits["train"]["Date"].max()
        assert train_last_date.year == 2021

        # Test should start in 2022
        test_first_date = splits["test"]["Date"].min()
        assert test_first_date.year == 2022

    def test_split_with_overlapping_ranges(self, sample_data):
        """Test splitting with overlapping date ranges."""
        # Overlapping ranges should still work (might be intentional for validation)
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-06-30"), "overlap": ("2021-01-01", "2021-12-31")})
        splits = splitter.split(sample_data)

        assert len(splits["train"]) > 0
        assert len(splits["overlap"]) > 0

    def test_split_with_gaps_in_ranges(self, sample_data):
        """Test splitting with gaps between ranges."""
        splitter = DateRangeSplitter(
            {
                "train": ("2020-01-01", "2020-12-31"),
                "test": ("2022-01-01", "2022-12-31"),  # Gap: all of 2021
            }
        )
        splits = splitter.split(sample_data)

        # Should only include data in specified ranges
        assert splits["train"]["Date"].max().year == 2020
        assert splits["test"]["Date"].min().year == 2022

    def test_split_empty_dataframe_raises_error(self):
        """Test that splitting empty DataFrame raises ValueError."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31")})
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty DataFrame"):
            splitter.split(empty_df)

    def test_split_without_date_column_raises_error(self):
        """Test that DataFrame without date column raises ValueError."""
        df = pd.DataFrame({"value": range(100)})
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31")})

        with pytest.raises(ValueError, match="Date column not found"):
            splitter.split(df)

    def test_split_with_different_date_column_names(self):
        """Test splitting with various date column names."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")

        for date_col in ["Date", "date", "timestamp", "Timestamp"]:
            df = pd.DataFrame({date_col: dates, "value": range(len(dates))})

            splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
            splits = splitter.split(df)

            assert len(splits["train"]) == 731
            assert len(splits["test"]) == 365

    def test_split_with_timezone_aware_dates(self):
        """Test splitting with timezone-aware dates."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
        df = pd.DataFrame({"Date": dates, "value": range(len(dates))})

        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splits = splitter.split(df)

        # Should remove timezone
        assert splits["train"]["Date"].dt.tz is None
        assert splits["test"]["Date"].dt.tz is None

    def test_split_with_unsorted_dates(self, sample_data):
        """Test that splitter sorts data by date."""
        # Shuffle the data
        shuffled = sample_data.sample(frac=1, random_state=42).reset_index(drop=True)

        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splits = splitter.split(shuffled)

        # Verify splits are sorted
        assert splits["train"]["Date"].is_monotonic_increasing
        assert splits["test"]["Date"].is_monotonic_increasing

    def test_split_with_no_matching_data(self, sample_data):
        """Test split when date range has no matching data."""
        # Request data from 2030 when data only goes to 2022
        splitter = DateRangeSplitter({"future": ("2030-01-01", "2030-12-31")})
        splits = splitter.split(sample_data)

        # Should return empty DataFrame for this range
        assert len(splits["future"]) == 0

    def test_split_with_partial_overlap(self, sample_data):
        """Test split with date range that partially overlaps data."""
        # Request range from 2021-2023, but data ends in 2022
        splitter = DateRangeSplitter({"partial": ("2021-01-01", "2023-12-31")})
        splits = splitter.split(sample_data)

        # Should only include 2021-2022 data
        assert splits["partial"]["Date"].min().year == 2021
        assert splits["partial"]["Date"].max().year == 2022


class TestDateRangeSplitterMetadata:
    """Test DateRangeSplitter.get_metadata() method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        return pd.DataFrame({"Date": dates, "Open": range(len(dates)), "Close": range(50, 50 + len(dates))})

    def test_metadata_before_split(self):
        """Test metadata before calling split()."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        metadata = splitter.get_metadata()

        assert metadata["type"] == "date_range"
        assert "ranges" in metadata
        assert metadata["ranges"]["train"] == ("2020-01-01", "2021-12-31")

    def test_metadata_after_split(self, sample_data):
        """Test metadata after calling split()."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        assert metadata["type"] == "date_range"
        assert "date_ranges" in metadata
        assert "final_shapes" in metadata

    def test_metadata_date_ranges(self, sample_data):
        """Test that metadata contains correct date ranges."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        assert metadata["date_ranges"]["train"]["start"] == "2020-01-01"
        assert metadata["date_ranges"]["train"]["end"] == "2021-12-31"
        assert metadata["date_ranges"]["test"]["start"] == "2022-01-01"
        assert metadata["date_ranges"]["test"]["end"] == "2022-12-31"

    def test_metadata_shapes(self, sample_data):
        """Test that metadata contains correct shapes."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        assert metadata["final_shapes"]["train"] == (731, 3)  # 2 years (1 leap)
        assert metadata["final_shapes"]["test"] == (365, 3)  # 1 year

    def test_metadata_with_empty_split(self, sample_data):
        """Test metadata when split has no matching data."""
        splitter = DateRangeSplitter({"future": ("2030-01-01", "2030-12-31")})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        # Should still have metadata even for empty split
        assert "final_shapes" in metadata
        assert metadata["final_shapes"]["future"][0] == 0


class TestDateRangeSplitterProtocol:
    """Test that DateRangeSplitter adheres to DataSplitter protocol."""

    def test_implements_protocol(self):
        """Test that DateRangeSplitter implements DataSplitter
        protocol."""
        from quantrl_lab.data.partitioning import DataSplitter

        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31")})
        assert isinstance(splitter, DataSplitter)

    def test_has_split_method(self):
        """Test that splitter has split() method."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31")})
        assert hasattr(splitter, "split")
        assert callable(splitter.split)

    def test_has_get_metadata_method(self):
        """Test that splitter has get_metadata() method."""
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2021-12-31")})
        assert hasattr(splitter, "get_metadata")
        assert callable(splitter.get_metadata)

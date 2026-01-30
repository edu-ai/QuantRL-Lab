"""Tests for date parsing utility module."""

from datetime import datetime

import pandas as pd
import pytest

from quantrl_lab.data.utils.date_parsing import (
    format_date_to_string,
    normalize_date,
    normalize_date_range,
)


class TestNormalizeDate:
    """Tests for normalize_date function."""

    def test_normalize_string_date(self):
        """Test normalizing a string date."""
        result = normalize_date("2023-01-15")
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_normalize_datetime_object(self):
        """Test normalizing a datetime object returns same object."""
        dt = datetime(2023, 1, 15, 10, 30, 0)
        result = normalize_date(dt)
        assert result == dt
        assert isinstance(result, datetime)

    def test_normalize_pandas_timestamp(self):
        """Test normalizing a pandas Timestamp."""
        ts = pd.Timestamp("2023-01-15")
        result = normalize_date(ts)
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15

    def test_normalize_none_with_default(self):
        """Test normalizing None with default value."""
        default = datetime(2023, 1, 1)
        result = normalize_date(None, default_if_none=default)
        assert result == default

    def test_normalize_none_without_default_raises(self):
        """Test that normalizing None without default raises
        ValueError."""
        with pytest.raises(ValueError, match="Date cannot be None"):
            normalize_date(None)

    def test_normalize_invalid_string_raises(self):
        """Test that invalid date string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date string"):
            normalize_date("not a date")

    def test_normalize_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported date type"):
            normalize_date(12345)

    def test_normalize_iso_format(self):
        """Test normalizing ISO format date string."""
        result = normalize_date("2023-01-15T10:30:00")
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30


class TestNormalizeDateRange:
    """Tests for normalize_date_range function."""

    def test_normalize_valid_range(self):
        """Test normalizing a valid date range."""
        start, end = normalize_date_range("2023-01-01", "2023-12-31")
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end

    def test_normalize_range_end_defaults_to_now(self):
        """Test that end defaults to now if not provided."""
        start, end = normalize_date_range("2023-01-01")
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start < end

    def test_normalize_range_validates_order(self):
        """Test that start > end raises ValueError when
        validate_order=True."""
        with pytest.raises(ValueError, match="must be before or equal to"):
            normalize_date_range("2023-12-31", "2023-01-01", validate_order=True)

    def test_normalize_range_no_validation(self):
        """Test that validation can be disabled."""
        # Should not raise even though start > end
        start, end = normalize_date_range("2023-12-31", "2023-01-01", validate_order=False)
        assert start > end

    def test_normalize_range_none_start_raises(self):
        """Test that None start raises ValueError."""
        with pytest.raises(ValueError, match="Start date cannot be None"):
            normalize_date_range(None, "2023-12-31")

    def test_normalize_range_none_end_without_default_raises(self):
        """Test that None end without default_end_to_now raises
        ValueError."""
        with pytest.raises(ValueError, match="End date cannot be None"):
            normalize_date_range("2023-01-01", None, default_end_to_now=False)

    def test_normalize_range_with_datetime_objects(self):
        """Test normalizing with datetime objects."""
        start_dt = datetime(2023, 1, 1)
        end_dt = datetime(2023, 12, 31)
        start, end = normalize_date_range(start_dt, end_dt)
        assert start == start_dt
        assert end == end_dt

    def test_normalize_range_mixed_types(self):
        """Test normalizing with mixed types (string and datetime)."""
        start, end = normalize_date_range("2023-01-01", datetime(2023, 12, 31))
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start.year == 2023
        assert end.year == 2023


class TestFormatDateToString:
    """Tests for format_date_to_string function."""

    def test_format_datetime_default(self):
        """Test formatting datetime with default format."""
        dt = datetime(2023, 1, 15, 10, 30, 0)
        result = format_date_to_string(dt)
        assert result == "2023-01-15"

    def test_format_datetime_custom_format(self):
        """Test formatting datetime with custom format."""
        dt = datetime(2023, 1, 15, 10, 30, 0)
        result = format_date_to_string(dt, format_string="%Y/%m/%d %H:%M")
        assert result == "2023/01/15 10:30"

    def test_format_string_date(self):
        """Test formatting a string date (parses first)."""
        result = format_date_to_string("2023-01-15")
        assert result == "2023-01-15"

    def test_format_pandas_timestamp(self):
        """Test formatting a pandas Timestamp."""
        ts = pd.Timestamp("2023-01-15")
        result = format_date_to_string(ts)
        assert result == "2023-01-15"

    def test_format_with_time_components(self):
        """Test formatting preserves time when format includes it."""
        dt = datetime(2023, 1, 15, 14, 30, 45)
        result = format_date_to_string(dt, format_string="%Y-%m-%d %H:%M:%S")
        assert result == "2023-01-15 14:30:45"

    def test_format_invalid_date_raises(self):
        """Test that invalid date raises ValueError."""
        with pytest.raises(ValueError):
            format_date_to_string("invalid date")

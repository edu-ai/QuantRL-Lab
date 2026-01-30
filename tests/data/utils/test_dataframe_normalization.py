"""Tests for DataFrame normalization utility module."""

import pandas as pd
import pytest

from quantrl_lab.data.utils.dataframe_normalization import (
    add_date_column_from_timestamp,
    add_symbol_column,
    convert_columns_to_numeric,
    sort_by_timestamp,
    standardize_ohlcv_columns,
    standardize_ohlcv_dataframe,
)


class TestStandardizeOHLCVColumns:
    """Tests for standardize_ohlcv_columns function."""

    def test_rename_basic_columns(self):
        """Test renaming basic OHLCV columns."""
        df = pd.DataFrame(
            {
                'date': ['2023-01-01'],
                'open': [150.0],
                'close': [152.0],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open', 'close': 'Close'}
        result = standardize_ohlcv_columns(df, mapping)

        assert 'Timestamp' in result.columns
        assert 'Open' in result.columns
        assert 'Close' in result.columns
        assert 'date' not in result.columns

    def test_rename_with_unmapped_columns(self):
        """Test that unmapped columns are kept by default."""
        df = pd.DataFrame(
            {
                'date': ['2023-01-01'],
                'open': [150.0],
                'extra_col': ['data'],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open'}
        result = standardize_ohlcv_columns(df, mapping, drop_unmapped=False)

        assert 'extra_col' in result.columns

    def test_rename_drop_unmapped(self):
        """Test dropping unmapped columns."""
        df = pd.DataFrame(
            {
                'date': ['2023-01-01'],
                'open': [150.0],
                'extra_col': ['data'],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open'}
        result = standardize_ohlcv_columns(df, mapping, drop_unmapped=True)

        assert 'extra_col' not in result.columns
        assert 'Timestamp' in result.columns
        assert 'Open' in result.columns


class TestAddSymbolColumn:
    """Tests for add_symbol_column function."""

    def test_add_symbol_at_end(self):
        """Test adding Symbol column at end."""
        df = pd.DataFrame({'Open': [150.0], 'Close': [152.0]})
        result = add_symbol_column(df, 'AAPL', position='end')

        assert 'Symbol' in result.columns
        assert result['Symbol'].iloc[0] == 'AAPL'
        assert result.columns[-1] == 'Symbol'

    def test_add_symbol_at_start(self):
        """Test adding Symbol column at start."""
        df = pd.DataFrame({'Open': [150.0], 'Close': [152.0]})
        result = add_symbol_column(df, 'AAPL', position='start')

        assert 'Symbol' in result.columns
        assert result['Symbol'].iloc[0] == 'AAPL'
        assert result.columns[0] == 'Symbol'

    def test_add_symbol_multiple_rows(self):
        """Test adding Symbol to multiple rows."""
        df = pd.DataFrame({'Open': [150.0, 151.0], 'Close': [152.0, 153.0]})
        result = add_symbol_column(df, 'GOOGL')

        assert len(result['Symbol']) == 2
        assert all(result['Symbol'] == 'GOOGL')


class TestAddDateColumnFromTimestamp:
    """Tests for add_date_column_from_timestamp function."""

    def test_add_date_from_timestamp(self):
        """Test adding Date column from Timestamp."""
        df = pd.DataFrame({'Timestamp': pd.to_datetime(['2023-01-01 10:30:00', '2023-01-02 14:45:00'])})
        result = add_date_column_from_timestamp(df)

        assert 'Date' in result.columns
        assert len(result['Date']) == 2
        assert str(result['Date'].iloc[0]) == '2023-01-01'

    def test_add_date_converts_string_timestamp(self):
        """Test that string timestamps are converted to datetime
        first."""
        df = pd.DataFrame({'Timestamp': ['2023-01-01 10:30:00', '2023-01-02 14:45:00']})
        result = add_date_column_from_timestamp(df)

        assert 'Date' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['Timestamp'])

    def test_add_date_missing_timestamp_raises(self):
        """Test that missing Timestamp column raises ValueError."""
        df = pd.DataFrame({'Open': [150.0]})
        with pytest.raises(ValueError, match="Timestamp column .* not found"):
            add_date_column_from_timestamp(df)

    def test_add_date_custom_column_names(self):
        """Test using custom column names."""
        df = pd.DataFrame({'MyTime': pd.to_datetime(['2023-01-01'])})
        result = add_date_column_from_timestamp(df, timestamp_col='MyTime', date_col='MyDate')

        assert 'MyDate' in result.columns


class TestSortByTimestamp:
    """Tests for sort_by_timestamp function."""

    def test_sort_ascending(self):
        """Test sorting by timestamp in ascending order."""
        df = pd.DataFrame({'Timestamp': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02']), 'Value': [3, 1, 2]})
        result = sort_by_timestamp(df, ascending=True)

        assert result['Value'].tolist() == [1, 2, 3]

    def test_sort_descending(self):
        """Test sorting by timestamp in descending order."""
        df = pd.DataFrame({'Timestamp': pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-02']), 'Value': [1, 3, 2]})
        result = sort_by_timestamp(df, ascending=False)

        assert result['Value'].tolist() == [3, 2, 1]

    def test_sort_resets_index(self):
        """Test that index is reset after sorting."""
        df = pd.DataFrame({'Timestamp': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02'])})
        result = sort_by_timestamp(df, reset_index=True)

        assert result.index.tolist() == [0, 1, 2]

    def test_sort_no_reset_index(self):
        """Test sorting without resetting index."""
        df = pd.DataFrame({'Timestamp': pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02'])})
        result = sort_by_timestamp(df, reset_index=False)

        # Original indices should be preserved in order
        assert result.index.tolist() == [1, 2, 0]

    def test_sort_missing_column_raises(self):
        """Test that missing timestamp column raises ValueError."""
        df = pd.DataFrame({'Value': [1, 2, 3]})
        with pytest.raises(ValueError, match="Timestamp column .* not found"):
            sort_by_timestamp(df)


class TestConvertColumnsToNumeric:
    """Tests for convert_columns_to_numeric function."""

    def test_convert_specified_columns(self):
        """Test converting specified columns to numeric."""
        df = pd.DataFrame({'Open': ['150.0', '151.0'], 'Close': ['152.0', '153.0'], 'Symbol': ['AAPL', 'AAPL']})
        result = convert_columns_to_numeric(df, columns=['Open', 'Close'])

        assert pd.api.types.is_numeric_dtype(result['Open'])
        assert pd.api.types.is_numeric_dtype(result['Close'])
        assert not pd.api.types.is_numeric_dtype(result['Symbol'])

    def test_convert_auto_detect_ohlcv(self):
        """Test auto-detecting OHLCV columns."""
        df = pd.DataFrame(
            {
                'Open': ['150.0'],
                'High': ['155.0'],
                'Low': ['149.0'],
                'Close': ['152.0'],
                'Volume': ['1000000'],
            }
        )
        result = convert_columns_to_numeric(df, columns=None)

        assert pd.api.types.is_numeric_dtype(result['Open'])
        assert pd.api.types.is_numeric_dtype(result['Volume'])

    def test_convert_coerce_errors(self):
        """Test that invalid values are coerced to NaN."""
        df = pd.DataFrame({'Open': ['150.0', 'invalid', '152.0']})
        result = convert_columns_to_numeric(df, columns=['Open'], errors='coerce')

        assert pd.isna(result['Open'].iloc[1])
        assert result['Open'].iloc[0] == 150.0

    def test_convert_missing_column_ignored(self):
        """Test that missing columns are ignored gracefully."""
        df = pd.DataFrame({'Open': ['150.0']})
        # Should not raise even though Close is not present
        result = convert_columns_to_numeric(df, columns=['Open', 'Close'])
        assert 'Close' not in result.columns


class TestStandardizeOHLCVDataFrame:
    """Tests for standardize_ohlcv_dataframe function."""

    def test_full_pipeline(self):
        """Test the full standardization pipeline."""
        df = pd.DataFrame(
            {
                'date': ['2023-01-02', '2023-01-01'],
                'open': ['150.0', '148.0'],
                'close': ['152.0', '149.0'],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open', 'close': 'Close'}

        result = standardize_ohlcv_dataframe(
            df, column_mapping=mapping, symbol='AAPL', add_date=True, sort_data=True, convert_numeric=True
        )

        # Check column renaming
        assert 'Timestamp' in result.columns
        assert 'Open' in result.columns

        # Check symbol added
        assert 'Symbol' in result.columns
        assert all(result['Symbol'] == 'AAPL')

        # Check date added
        assert 'Date' in result.columns

        # Check sorted (should be 2023-01-01 first now)
        assert result['Open'].iloc[0] == 148.0

        # Check numeric conversion
        assert pd.api.types.is_numeric_dtype(result['Open'])

    def test_pipeline_without_symbol(self):
        """Test pipeline without adding symbol."""
        df = pd.DataFrame(
            {
                'date': ['2023-01-01'],
                'open': ['150.0'],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open'}

        result = standardize_ohlcv_dataframe(df, mapping, symbol=None)

        assert 'Symbol' not in result.columns

    def test_pipeline_skip_date(self):
        """Test pipeline skipping date addition."""
        df = pd.DataFrame(
            {
                'date': pd.to_datetime(['2023-01-01']),
                'open': ['150.0'],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open'}

        result = standardize_ohlcv_dataframe(df, mapping, add_date=False)

        assert 'Date' not in result.columns

    def test_pipeline_skip_sort(self):
        """Test pipeline skipping sorting."""
        df = pd.DataFrame(
            {
                'date': pd.to_datetime(['2023-01-02', '2023-01-01']),
                'value': [2, 1],
            }
        )
        mapping = {'date': 'Timestamp'}

        result = standardize_ohlcv_dataframe(df, mapping, sort_data=False)

        # Should remain in original order
        assert result['value'].iloc[0] == 2

    def test_pipeline_skip_numeric_conversion(self):
        """Test pipeline skipping numeric conversion."""
        df = pd.DataFrame(
            {
                'date': pd.to_datetime(['2023-01-01']),
                'open': ['150.0'],
            }
        )
        mapping = {'date': 'Timestamp', 'open': 'Open'}

        result = standardize_ohlcv_dataframe(df, mapping, convert_numeric=False)

        # Should remain as string
        assert result['Open'].dtype == object

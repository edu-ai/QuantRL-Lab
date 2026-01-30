"""Tests for symbol handling utility module."""

import pytest

from quantrl_lab.data.utils.symbol_handling import (
    get_single_symbol,
    normalize_symbols,
    validate_symbols,
)


class TestNormalizeSymbols:
    """Tests for normalize_symbols function."""

    def test_normalize_single_string(self):
        """Test normalizing a single string symbol."""
        result = normalize_symbols("AAPL")
        assert result == ["AAPL"]
        assert isinstance(result, list)

    def test_normalize_list(self):
        """Test normalizing a list of symbols."""
        result = normalize_symbols(["AAPL", "GOOGL", "MSFT"])
        assert result == ["AAPL", "GOOGL", "MSFT"]
        assert isinstance(result, list)

    def test_normalize_empty_list_does_not_raise(self):
        """Test that empty list is normalized to empty list."""
        # Empty list passes through normalize_symbols without error
        result = normalize_symbols([])
        assert result == []

    def test_normalize_non_string_in_list_raises(self):
        """Test that non-string elements raise ValueError."""
        with pytest.raises(ValueError, match="All elements in symbols list must be strings"):
            normalize_symbols(["AAPL", 123, "GOOGL"])

    def test_normalize_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be a string or list of strings"):
            normalize_symbols(12345)

    def test_normalize_with_max_symbols(self):
        """Test normalizing with max_symbols limit."""
        result = normalize_symbols(["AAPL", "GOOGL", "MSFT"], max_symbols=2)
        assert len(result) == 2
        assert result == ["AAPL", "GOOGL"]

    def test_normalize_max_symbols_no_warning(self):
        """Test that max_symbols without exceeding doesn't warn."""
        result = normalize_symbols(["AAPL", "GOOGL"], max_symbols=5)
        assert len(result) == 2
        assert result == ["AAPL", "GOOGL"]

    def test_normalize_single_symbol_list(self):
        """Test normalizing a single-element list."""
        result = normalize_symbols(["AAPL"])
        assert result == ["AAPL"]


class TestValidateSymbols:
    """Tests for validate_symbols function."""

    def test_validate_single_string(self):
        """Test validating a single string symbol."""
        # Should not raise
        validate_symbols("AAPL")

    def test_validate_list(self):
        """Test validating a list of symbols."""
        # Should not raise
        validate_symbols(["AAPL", "GOOGL"])

    def test_validate_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            validate_symbols([])

    def test_validate_non_string_raises(self):
        """Test that non-string element raises ValueError."""
        with pytest.raises(ValueError, match="not a string"):
            validate_symbols(["AAPL", 123])

    def test_validate_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be a string or list"):
            validate_symbols(12345)

    def test_validate_empty_string_raises(self):
        """Test that empty string raises ValueError by default."""
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            validate_symbols("")

    def test_validate_empty_string_allowed(self):
        """Test that empty string is allowed with allow_empty=True."""
        # Should not raise
        validate_symbols("", allow_empty=True)

    def test_validate_whitespace_string_raises(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            validate_symbols("   ")

    def test_validate_max_symbols(self):
        """Test validating with max_symbols limit."""
        with pytest.raises(ValueError, match="Too many symbols"):
            validate_symbols(["AAPL", "GOOGL", "MSFT"], max_symbols=2)

    def test_validate_within_max_symbols(self):
        """Test that validation passes when within limit."""
        # Should not raise
        validate_symbols(["AAPL", "GOOGL"], max_symbols=5)


class TestGetSingleSymbol:
    """Tests for get_single_symbol function."""

    def test_get_from_string(self):
        """Test getting symbol from single string."""
        result = get_single_symbol("AAPL")
        assert result == "AAPL"
        assert isinstance(result, str)

    def test_get_from_single_element_list(self):
        """Test getting symbol from single-element list."""
        result = get_single_symbol(["AAPL"])
        assert result == "AAPL"

    def test_get_from_multi_element_list(self):
        """Test getting first symbol from multi-element list."""
        result = get_single_symbol(["AAPL", "GOOGL", "MSFT"])
        assert result == "AAPL"

    def test_get_from_multi_element_list_no_warn(self):
        """Test getting from multi-element list without warning."""
        result = get_single_symbol(["AAPL", "GOOGL"], warn_on_multiple=False)
        assert result == "AAPL"

    def test_get_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError):
            get_single_symbol(12345)

    def test_get_from_empty_list_raises(self):
        """Test that empty list raises IndexError."""
        with pytest.raises(IndexError):
            get_single_symbol([])

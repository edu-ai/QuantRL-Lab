"""
Data splitting strategies for train/test/validation splits.

This module provides protocol-based data splitters that can split
DataFrames by ratio or date ranges for time series data.
"""

from .base import DataSplitter
from .date_range import DateRangeSplitter
from .ratio import RatioSplitter

__all__ = [
    "DataSplitter",
    "DateRangeSplitter",
    "RatioSplitter",
]

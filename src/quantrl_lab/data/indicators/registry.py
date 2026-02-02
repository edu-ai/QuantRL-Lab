from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import pandas as pd


@dataclass
class IndicatorMetadata:
    """
    Metadata for registered indicators.

    Attributes:
        name: Indicator name (e.g., 'SMA', 'RSI')
        func: The callable function that computes the indicator
        required_columns: Set of required DataFrame columns (e.g., {'close', 'volume'})
        output_columns: List of column names this indicator adds to DataFrame
        dependencies: List of other indicator names that must be computed first
        description: Human-readable description of what the indicator computes
    """

    name: str
    func: Callable
    required_columns: Set[str] = field(default_factory=set)
    output_columns: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        """Auto-generate output_columns if not provided."""
        if not self.output_columns:
            self.output_columns = [self.name]


class IndicatorRegistry:
    """
    Registry for technical indicators with metadata and validation.

    This registry uses a decorator pattern to register indicator functions
    along with metadata about their requirements and outputs. It provides
    validation to ensure DataFrames have the required columns before applying
    indicators.

    Example:
        >>> @IndicatorRegistry.register(
        ...     name='SMA',
        ...     required_columns={'close'},
        ...     output_columns=['SMA'],
        ...     description="Simple Moving Average"
        ... )
        ... def sma(df, window=20, column='close'):
        ...     df[f'SMA_{window}'] = df[column].rolling(window=window).mean()
        ...     return df
        >>>
        >>> # Use with validation
        >>> df = IndicatorRegistry.apply_safe('SMA', df, window=20)
    """

    # Mapping of indicator names to metadata objects
    _indicators: Dict[str, IndicatorMetadata] = {}

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        required_columns: Optional[Set[str]] = None,
        output_columns: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ) -> Callable:
        """
        Register an indicator function with metadata.

        Args:
            name: Indicator name. If None, uses function name.
            required_columns: Set of required DataFrame columns (case-insensitive).
                Example: {'close'}, {'high', 'low', 'close'}
            output_columns: List of column names this indicator will add.
                If None, defaults to [name].
            dependencies: List of other indicator names that must be applied first.
            description: Human-readable description of the indicator.

        Returns:
            Decorator function that registers the indicator.

        Example:
            >>> @IndicatorRegistry.register(
            ...     name='RSI',
            ...     required_columns={'close'},
            ...     output_columns=['RSI'],
            ...     description="Relative Strength Index"
            ... )
            ... def rsi(df, window=14):
            ...     # calculation
            ...     return df
        """

        def decorator(func: Callable):
            indicator_name = name or func.__name__

            metadata = IndicatorMetadata(
                name=indicator_name,
                func=func,
                required_columns=required_columns or set(),
                output_columns=output_columns or [indicator_name],
                dependencies=dependencies or [],
                description=description,
            )

            cls._indicators[indicator_name] = metadata
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get the indicator function by name.

        Args:
            name: Name of the indicator

        Raises:
            KeyError: If the name is not found in the registry

        Returns:
            Callable: Indicator function
        """
        if name not in cls._indicators:
            raise KeyError(f"Indicator '{name}' not registered")
        return cls._indicators[name].func

    @classmethod
    def get_metadata(cls, name: str) -> IndicatorMetadata:
        """
        Get the full metadata for an indicator.

        Args:
            name: Name of the indicator

        Raises:
            KeyError: If the name is not found in the registry

        Returns:
            IndicatorMetadata: Metadata object for the indicator
        """
        if name not in cls._indicators:
            raise KeyError(f"Indicator '{name}' not registered")
        return cls._indicators[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered indicators.

        Returns:
            List[str]: List of indicator names
        """
        return list(cls._indicators.keys())

    @classmethod
    def validate_compatibility(cls, df: pd.DataFrame, indicator_name: str) -> bool:
        """
        Check if DataFrame has required columns for indicator.

        Performs case-insensitive column checking.

        Args:
            df: DataFrame to validate
            indicator_name: Name of the indicator to check

        Returns:
            bool: True if DataFrame has all required columns

        Raises:
            KeyError: If indicator is not registered
        """
        if indicator_name not in cls._indicators:
            raise KeyError(f"Indicator '{indicator_name}' not registered")

        metadata = cls._indicators[indicator_name]

        # Case-insensitive column check
        df_columns_lower = {col.lower() for col in df.columns}
        required_lower = {col.lower() for col in metadata.required_columns}

        return required_lower.issubset(df_columns_lower)

    @classmethod
    def get_missing_columns(cls, df: pd.DataFrame, indicator_name: str) -> Set[str]:
        """
        Get the set of missing required columns for an indicator.

        Args:
            df: DataFrame to check
            indicator_name: Name of the indicator

        Returns:
            Set[str]: Set of missing column names (from required_columns)

        Raises:
            KeyError: If indicator is not registered
        """
        if indicator_name not in cls._indicators:
            raise KeyError(f"Indicator '{indicator_name}' not registered")

        metadata = cls._indicators[indicator_name]

        # Case-insensitive column check
        df_columns_lower = {col.lower() for col in df.columns}
        required_lower = {col.lower() for col in metadata.required_columns}

        missing = required_lower - df_columns_lower

        # Map back to original case from metadata
        result = set()
        for req_col in metadata.required_columns:
            if req_col.lower() in missing:
                result.add(req_col)

        return result

    @classmethod
    def apply(cls, name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply the indicator function to the dataframe.

        Args:
            name: Name of the indicator
            df: Input dataframe
            **kwargs: Additional keyword arguments to be passed to the indicator function

        Returns:
            pd.DataFrame: DataFrame with the indicator added

        Raises:
            KeyError: If indicator is not registered
        """
        indicator_func = cls.get(name)
        return indicator_func(df, **kwargs)

    @classmethod
    def apply_safe(cls, name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply indicator with validation.

        Validates that the DataFrame has all required columns before applying
        the indicator. Raises a descriptive error if columns are missing.

        Args:
            name: Name of the indicator
            df: Input dataframe
            **kwargs: Additional keyword arguments to be passed to the indicator function

        Returns:
            pd.DataFrame: DataFrame with the indicator added

        Raises:
            KeyError: If indicator is not registered
            ValueError: If DataFrame is missing required columns
        """
        if not cls.validate_compatibility(df, name):
            missing = cls.get_missing_columns(df, name)
            raise ValueError(
                f"Cannot apply indicator '{name}': missing required columns {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        return cls.apply(name, df, **kwargs)

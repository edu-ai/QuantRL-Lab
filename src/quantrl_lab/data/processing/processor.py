import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from rich.console import Console

# Import centralized configuration
from quantrl_lab.data.config import config
from quantrl_lab.data.partitioning import DateRangeSplitter, RatioSplitter
from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator

# Import new feature generators
from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator

# Import sentiment modules
from quantrl_lab.data.processing.sentiment import (
    HuggingFaceProvider,
    SentimentConfig,
)

console: Console = Console()


@dataclass
class ProcessingMetadata:
    """
    Metadata collected during data processing pipeline.

    This dataclass tracks all transformations and operations applied during
    the data processing pipeline, providing transparency and reproducibility.

    Attributes:
        symbol (Optional[Union[str, List[str]]]): Stock symbol(s) being processed.
            Single symbol as string, multiple as list.
        date_ranges (Dict[str, Dict[str, str]]): Date ranges for each data split.
            Format: {"split_name": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
        fillna_strategy (str): Strategy used for filling missing sentiment scores.
            Options: "neutral" (fill with 0.0) or "fill_forward" (forward fill)
        technical_indicators (List[Union[str, Dict]]): List of technical indicators applied.
            Can contain strings ("SMA") or dicts ({"SMA": {"window": 20}})
        news_sentiment_applied (bool): Whether news sentiment analysis was performed.
        columns_dropped (List[str]): List of columns dropped during processing.
        original_shape (Tuple[int, int]): Shape of input data before processing (rows, cols).
        final_shapes (Dict[str, Tuple[int, int]]): Shapes of output data after processing.
            Format: {"split_name": (rows, cols)} or {"full_data": (rows, cols)}

    Examples:
        >>> metadata = ProcessingMetadata(
        ...     symbol="AAPL",
        ...     fillna_strategy="neutral",
        ...     original_shape=(1000, 7)
        ... )
        >>> metadata.technical_indicators = ["SMA", "RSI"]
        >>> metadata.to_dict()
    """

    symbol: Optional[Union[str, List[str]]] = None
    date_ranges: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fillna_strategy: str = "neutral"
    technical_indicators: List[Union[str, Dict]] = field(default_factory=list)
    news_sentiment_applied: bool = False
    analyst_data_applied: bool = False
    market_context_applied: bool = False
    columns_dropped: List[str] = field(default_factory=list)
    original_shape: Tuple[int, int] = (0, 0)
    final_shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """
        Convert metadata to dictionary format for backward
        compatibility.

        Returns:
            Dict: Dictionary representation of metadata with all fields

        Examples:
            >>> metadata = ProcessingMetadata(symbol="AAPL", original_shape=(100, 5))
            >>> result = metadata.to_dict()
            >>> assert result["symbol"] == "AAPL"
            >>> assert result["original_shape"] == (100, 5)
        """
        return {
            "symbol": self.symbol,
            "date_ranges": self.date_ranges,
            "fillna_strategy": self.fillna_strategy,
            "technical_indicators": self.technical_indicators,
            "news_sentiment_applied": self.news_sentiment_applied,
            "analyst_data_applied": self.analyst_data_applied,
            "market_context_applied": self.market_context_applied,
            "columns_dropped": self.columns_dropped,
            "original_shape": self.original_shape,
            "final_shapes": self.final_shapes,
        }


class DataProcessor:
    @staticmethod
    def load_indicators(file_path: str) -> List[Union[str, Dict]]:
        """
        Load indicator configuration from a YAML or JSON file.

        Args:
            file_path: Path to the configuration file (.yaml, .yml, or .json)

        Returns:
            List[Union[str, Dict]]: List of indicator configurations

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is unsupported or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            with open(file_path, "r") as f:
                if ext in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif ext == ".json":
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {ext}. Use .yaml or .json")

            # Validate structure - expect a list or a dict with an 'indicators' key
            if isinstance(config_data, list):
                return config_data
            elif isinstance(config_data, dict) and "indicators" in config_data:
                return config_data["indicators"]
            else:
                raise ValueError("Invalid config structure. Expected a list or a dict with 'indicators' key.")

        except Exception as e:
            raise ValueError(f"Failed to load indicator config from {file_path}: {e}")

    def __init__(self, ohlcv_data: pd.DataFrame, **kwargs):
        if ohlcv_data is None:
            raise ValueError("Required parameter 'ohlcv_data' is missing.")

        self.ohlcv_data = ohlcv_data  # minimal required data

        # === Optional data sources ===
        self.news_data = kwargs.get("news_data", None)
        self.fundamental_data = kwargs.get("fundamental_data", None)
        self.macro_data = kwargs.get("macro_data", None)
        self.calendar_event_data = kwargs.get("calendar_event_data", None)

        # === Analyst & Context Data ===
        self.analyst_grades = kwargs.get("analyst_grades", None)
        self.analyst_ratings = kwargs.get("analyst_ratings", None)
        self.sector_performance = kwargs.get("sector_performance", None)
        self.industry_performance = kwargs.get("industry_performance", None)

        # === Sentiment configuration and provider ===
        self.sentiment_config = kwargs.get("sentiment_config", SentimentConfig())
        self.sentiment_provider = kwargs.get("sentiment_provider")

        if self.sentiment_provider is None and self.news_data is not None:
            # Default to HuggingFaceProvider if news data is present but no provider given
            self.sentiment_provider = HuggingFaceProvider()

    def append_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[Union[str, Dict]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Add technical indicators to existing OHLCV DataFrame.

        Args:
            df (pd.DataFrame): raw OHLCV data
            indicators (Optional[List[Union[str, Dict]]], optional): Defaults to None.

        Raises:
            ValueError: if input DataFrame is empty
            ValueError: if required columns are missing

        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Return original if no indicators specified
        if not indicators:
            return df.copy()

        try:
            generator = TechnicalFeatureGenerator(indicators)
            return generator.generate(df, **kwargs)
        except ValueError as e:
            # Re-raise with same message or log
            raise e
        except Exception as e:
            console.print(f"[red]❌ Failed to append technical indicators: {e}[/red]")
            return df.copy()

    def append_news_sentiment_data(self, df: pd.DataFrame, fillna_strategy="neutral") -> pd.DataFrame:
        """
        Append news sentiment data to the OHLCV DataFrame.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame.
            fillna_strategy (str, optional): Strategy for handling missing sentiment scores. Defaults to "neutral".

        Raises:
            ValueError: If the input DataFrame is empty or if the strategy is unsupported.

        Returns:
            pd.DataFrame: DataFrame with appended news sentiment data.
        """
        if self.news_data is None or self.news_data.empty:
            console.print("[yellow]⚠️  No news data provided. Skipping sentiment analysis.[/yellow]")
            return df

        try:
            generator = SentimentFeatureGenerator(
                self.sentiment_provider, self.sentiment_config, self.news_data, fillna_strategy
            )
            return generator.generate(df)
        except ValueError as e:
            raise e
        except Exception as e:
            console.print(f"[red]❌ Failed to append sentiment data: {e}[/red]")
            return df

    def drop_unwanted_columns(
        self, df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None, keep_date: bool = False
    ) -> pd.DataFrame:
        """
        Drop unwanted columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns_to_drop (Optional[List[str]], optional): List of column names to drop.
                If None, will drop default columns ('Date', 'Timestamp', 'Symbol'). Defaults to None.
            keep_date (bool): If True, date-related columns will not be dropped.
        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """
        if columns_to_drop is None:
            columns_to_drop = [config.DEFAULT_DATE_COLUMN, "Timestamp", "Symbol"]
        elif not isinstance(columns_to_drop, list):
            raise ValueError(
                f"Invalid type for 'columns_to_drop': expected list, got {type(columns_to_drop).__name__}."
            )

        if keep_date:
            columns_to_drop = [col for col in columns_to_drop if col not in config.DATE_COLUMNS]

        return df.drop(columns=columns_to_drop, errors="ignore")

    def convert_columns_to_numeric(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert specified columns to numeric, handling date columns
        carefully.

        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[List[str]]): Specific columns to convert. If None, converts all object columns.

        Returns:
            pd.DataFrame: DataFrame with numeric conversions applied
        """
        if columns is None:
            # Only convert object columns that are not date-like
            columns = []
            for col in df.columns:
                if df[col].dtype == "object":
                    # Skip columns that look like dates
                    if col in config.DATE_COLUMNS or col.lower() in [c.lower() for c in config.DATE_COLUMNS]:
                        continue
                    # Check if it's actually a date column by looking at the data
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_val is not None:
                        try:
                            pd.to_datetime(sample_val)
                            # If conversion succeeds, it's probably a date column - skip it
                            continue
                        except (ValueError, TypeError):
                            # Not a date, safe to convert to numeric
                            columns.append(col)
        elif not isinstance(columns, list):
            raise ValueError(f"Invalid type for 'columns': expected list, got {type(columns).__name__}.")

        for col in columns:
            if col in df.columns and df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def data_processing_pipeline(
        self,
        indicators: Optional[List[Union[str, Dict]]] = None,
        fillna_strategy: str = "neutral",
        split_config: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]:
        """
        Main data processing pipeline.

        Applies technical indicators, sentiment analysis, and data transformations

        This method uses the DataPipeline infrastructure internally.

        Args:
            indicators (Optional[List[Union[str, Dict]]], optional):
                List of indicators to apply. Supports:
                - String format: ["SMA", "RSI"]
                - Dict format: [{"SMA": {"window": 20}}, {"RSI": {"window": 14}}]
                Defaults to None (no indicators).
            fillna_strategy (str, optional): Strategy for handling missing sentiment scores.
                Options: "neutral" (fill with 0.0) or "fill_forward" (forward fill).
                Defaults to "neutral".
            split_config (Optional[Dict], optional): Configuration for data splitting.
                If None, returns a single DataFrame. Otherwise, returns dict of DataFrames.
                Ratio-based: {'train': 0.7, 'test': 0.3}
                Date-based: {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}
            **kwargs: Additional arguments:
                - columns_to_drop: List of columns to drop (overrides default)
                - columns_to_convert: List of columns to convert to numeric

        Returns:
            Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]: A tuple containing:
                - Processed DataFrame if split_config is None
                - Dictionary of DataFrames if split_config is provided (keys: split names)
                - Metadata dictionary with processing information
        """
        from quantrl_lab.data.processing.pipeline import DataPipeline
        from quantrl_lab.data.processing.steps import (
            AnalystEstimatesStep,
            ColumnCleanupStep,
            MarketContextStep,
            NumericConversionStep,
            SentimentEnrichmentStep,
            TechnicalIndicatorStep,
            TimeFeatureStep,
        )

        # Build pipeline
        pipeline = DataPipeline()

        # 1. Technical Indicators
        pipeline.add_step(TechnicalIndicatorStep(indicators=indicators))

        # 2. Analyst Estimates
        if self.analyst_grades is not None or self.analyst_ratings is not None:
            pipeline.add_step(AnalystEstimatesStep(grades_df=self.analyst_grades, ratings_df=self.analyst_ratings))

        # 3. Market Context
        if self.sector_performance is not None or self.industry_performance is not None:
            pipeline.add_step(
                MarketContextStep(sector_perf_df=self.sector_performance, industry_perf_df=self.industry_performance)
            )

        # 4. Sentiment Enrichment (only if news data available)
        if self.news_data is not None:
            pipeline.add_step(
                SentimentEnrichmentStep(
                    news_data=self.news_data,
                    provider=self.sentiment_provider,
                    config=self.sentiment_config,
                    fillna_strategy=fillna_strategy,
                )
            )

        # 5. Numeric Conversion
        # Convert specified columns to numeric
        columns_to_convert = kwargs.get("columns_to_convert", None)
        pipeline.add_step(NumericConversionStep(columns=columns_to_convert))

        # 6. Time Features
        # Add cyclical time features (sin/cos) before cleanup drops the date column
        pipeline.add_step(TimeFeatureStep())

        # 7. Column Cleanup
        # If columns_to_drop is passed, use it; otherwise rely on defaults in step
        # Note: We keep date columns if splitting is required later
        columns_to_drop = kwargs.get("columns_to_drop", None)
        # If splitting, we MUST keep date columns for the split operation
        # If not splitting, the pipeline step handles default date dropping unless overridden
        keep_date = split_config is not None

        # Configure Cleanup Step
        cleanup_step = ColumnCleanupStep(columns_to_drop=columns_to_drop, keep_date=keep_date)
        pipeline.add_step(cleanup_step)

        # Execute Pipeline
        # We pass symbol for metadata tracking if available
        symbol = None
        if "Symbol" in self.ohlcv_data.columns:
            unique_symbols = self.ohlcv_data["Symbol"].unique()
            symbol = unique_symbols[0] if len(unique_symbols) == 1 else None

        processed_data, metadata_obj = pipeline.execute(self.ohlcv_data, symbol=symbol)

        # Update metadata flags
        if self.analyst_grades is not None or self.analyst_ratings is not None:
            metadata_obj.analyst_data_applied = True
        if self.sector_performance is not None or self.industry_performance is not None:
            metadata_obj.market_context_applied = True

        # Handle Data Splitting (Post-Processing)
        # Debug: Check for columns with all NaN values before dropna
        verbose = kwargs.get('verbose', False)
        if verbose:
            null_counts = processed_data.isnull().sum()
            all_null_cols = null_counts[null_counts == len(processed_data)]
            if not all_null_cols.empty:
                console.print(f"[yellow]⚠️  Warning: Columns with all NaN values: {list(all_null_cols.index)}[/yellow]")

            console.print(f"[cyan]Before dropna: {len(processed_data)} rows[/cyan]")
            console.print(f"[cyan]Columns in DataFrame: {list(processed_data.columns)}[/cyan]")

        # Drop rows with any NaN values
        # This handles:
        # 1. Indicator warm-up periods (e.g., SMA(200) creates 200 leading NaNs)
        # 2. Missing price data
        # 3. Any other features that couldn't be computed/filled
        initial_len = len(processed_data)
        processed_data = processed_data.dropna().reset_index(drop=True)
        dropped_count = initial_len - len(processed_data)

        if verbose:
            if dropped_count > 0:
                console.print(f"[yellow]Dropped {dropped_count} rows containing NaNs (indicator warm-up, etc)[/yellow]")
            else:
                console.print("[green]No rows dropped (data is clean)[/green]")

        if verbose:
            console.print(f"[cyan]After dropna: {len(processed_data)} rows[/cyan]")

        if split_config:
            split_data, split_metadata = self._split_data(processed_data, split_config)

            # Merge split metadata into pipeline metadata
            metadata_obj.date_ranges = split_metadata["date_ranges"]
            metadata_obj.final_shapes = split_metadata["final_shapes"]

            # Drop date column after splitting if it wasn't supposed to be kept
            for key in split_data:
                # Re-run cleanup to drop date columns now that splitting is done
                # unless user explicitly asked to keep them via columns_to_drop logic?
                # For safety, we replicate old behavior: drop defaults
                split_data[key] = self.drop_unwanted_columns(
                    split_data[key], [config.DEFAULT_DATE_COLUMN, "Timestamp", "Symbol"]
                )

            return split_data, metadata_obj.to_dict()
        else:
            # Handle metadata for non-split data (legacy logic port)
            date_column = next((col for col in config.DATE_COLUMNS if col in processed_data.columns), None)
            if date_column:
                dates = pd.to_datetime(processed_data[date_column])
                metadata_obj.date_ranges["full_data"] = {
                    "start": dates.min().strftime("%Y-%m-%d"),
                    "end": dates.max().strftime("%Y-%m-%d"),
                }
            metadata_obj.final_shapes["full_data"] = processed_data.shape

            # If we didn't split, we might still need to drop the date column if it was kept
            if not keep_date:
                pass

            return processed_data, metadata_obj.to_dict()

    def _split_data(self, df: pd.DataFrame, split_config: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Split the data into respective sets according to the config.

        This method now delegates to the new splitter classes (RatioSplitter or DateRangeSplitter)
        while maintaining backward compatibility with the existing API.

        Args:
            df (pd.DataFrame): input dataframe
            split_config (Dict): split config in dictionary format
                Example by ratio: {'train': 0.7, 'test': 0.3}
                Example by dates: {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}

        Raises:
            ValueError: If date column not found for splitting or invalid config

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict]: datasets in dict and metadata
        """
        # Determine split type based on config values
        is_date_based = any(isinstance(v, (tuple, list)) for v in split_config.values())

        if is_date_based:
            # Use DateRangeSplitter
            splitter = DateRangeSplitter(split_config)
        else:
            # Use RatioSplitter
            splitter = RatioSplitter(split_config)

        # Perform split
        split_data = splitter.split(df)

        # Get metadata from splitter
        metadata = splitter.get_metadata()

        return split_data, metadata

"""
Shared data-loading utilities for end-to-end examples.

This module centralises the data-acquisition and feature-engineering logic
that is common to both the single-stock and multi-symbol training pipelines:

  - ``init_data_sources``          – Initialise YFinance / FMP / Alpaca
  - ``get_date_range``             – Compute (start_date, end_date) from a lookback
  - ``select_alpha_indicators``    – Union-AlphaSelector across ≥1 symbol DataFrames
  - ``build_pipeline_for_symbol``  – Construct a DataPipeline from enrichment data
  - ``process_symbol``             – Apply pipeline + dropna for a single symbol
  - ``train_test_split_by_date``   – Temporal 80/20 split by unique calendar dates
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Data-source bundle
# ---------------------------------------------------------------------------


@dataclass
class DataSources:
    """
    Container for the initialised data-source clients used across
    examples.

    Attributes:
        loader:  YFinanceDataLoader (always available, no key required).
        fmp:     FMPDataSource or None (requires ``FMP_API_KEY`` in env).
        alpaca:  AlpacaDataLoader or None (requires ``ALPACA_API_KEY`` in env).
    """

    loader: object  # YFinanceDataLoader
    fmp: Optional[object] = field(default=None)  # FMPDataSource | None
    alpaca: Optional[object] = field(default=None)  # AlpacaDataLoader | None


def init_data_sources() -> DataSources:
    """
    Initialise and return all available data-source clients.

    YFinance is always initialised.  FMP and Alpaca are initialised only when
    the corresponding API key environment variables are set; a warning is
    printed when a key is missing.

    Returns:
        DataSources with ``loader``, ``fmp`` (optional), and ``alpaca`` (optional).
    """
    from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader

    loader = YFinanceDataLoader()

    fmp = None
    if os.environ.get("FMP_API_KEY"):
        try:
            from quantrl_lab.data.sources.fmp_loader import FMPDataSource

            fmp = FMPDataSource()
            console.print("[bold green]FMP connected — analyst & market-context data available.[/bold green]")
        except Exception as exc:
            console.print(f"[yellow]FMP connection failed: {exc} — skipping enrichment.[/yellow]")
    else:
        console.print("[dim]FMP_API_KEY not set — skipping analyst & market-context data.[/dim]")

    alpaca = None
    if os.environ.get("ALPACA_API_KEY"):
        try:
            from quantrl_lab.data.sources.alpaca_loader import AlpacaDataLoader

            alpaca = AlpacaDataLoader()
            console.print("[bold green]Alpaca connected — news-sentiment data available.[/bold green]")
        except Exception as exc:
            console.print(f"[yellow]Alpaca connection failed: {exc} — skipping news sentiment.[/yellow]")
    else:
        console.print("[dim]ALPACA_API_KEY not set — skipping news-sentiment data.[/dim]")

    return DataSources(loader=loader, fmp=fmp, alpaca=alpaca)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def get_date_range(period_years: float = 2.0) -> Tuple[datetime, datetime]:
    """
    Return ``(start_date, end_date)`` for a trailing lookback window.

    Args:
        period_years: Number of years to look back from today. Defaults to 2.

    Returns:
        Tuple of (start_date, end_date) as timezone-naive ``datetime`` objects.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * period_years))
    return start_date, end_date


# ---------------------------------------------------------------------------
# Alpha indicator selection
# ---------------------------------------------------------------------------


def select_alpha_indicators(
    raw_data: Dict[str, pd.DataFrame],
    *,
    metric: str = "ic",
    threshold: float = 0.015,
    top_k: int = 2,
    verbose: bool = False,
) -> List[Dict]:
    """
    Select technical indicators via AlphaSelector across ≥1 symbol
    DataFrames.

    Runs ``AlphaSelector.suggest_indicators`` on each symbol's raw OHLCV
    DataFrame and returns the *deduplicated union* of all suggested indicators.
    This "union" approach ensures that any alpha-generating signal found for
    any individual symbol is included in the shared feature set.

    Args:
        raw_data:  Mapping of ``{symbol: ohlcv_df}`` (raw, unprocessed).
        metric:    Scoring metric passed to ``AlphaSelector`` (e.g. ``"ic"``).
        threshold: Minimum metric value to keep an indicator.
        top_k:     Maximum indicators to pick *per symbol* before deduplication.
        verbose:   Whether to print per-symbol progress.

    Returns:
        Deduplicated list of indicator spec dicts, e.g.
        ``[{"name": "RSI", "params": {"window": 14}}, ...]``.
    """
    from quantrl_lab.alpha_research import AlphaSelector

    all_suggested: List[Dict] = []
    for symbol, df in raw_data.items():
        selector = AlphaSelector(df, verbose=verbose)
        indicators = selector.suggest_indicators(metric=metric, threshold=threshold, top_k=top_k)
        if verbose:
            console.print(f"  [cyan]{symbol}[/cyan]: {indicators}")
        all_suggested.extend(indicators)

    # Deduplicate while preserving insertion order
    seen: set = set()
    unique_indicators: List[Dict] = []
    for indicator in all_suggested:
        key = json.dumps(indicator, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_indicators.append(indicator)

    console.print(
        f"[green]Alpha selection: {len(unique_indicators)} unique indicators "
        f"across {len(raw_data)} symbol(s).[/green]"
    )
    return unique_indicators


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline_for_symbol(
    indicators: List[Dict],
    *,
    ratings_df: Optional[pd.DataFrame] = None,
    sector_perf_df: Optional[pd.DataFrame] = None,
    industry_perf_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
) -> "DataPipeline":  # noqa: F821  (forward ref to avoid circular import)
    """
    Build a ``DataPipeline`` for a single symbol from pre-fetched
    enrichment data.

    The pipeline always includes:
    1. ``TechnicalIndicatorStep`` (with the supplied indicator specs).
    2. ``NumericConversionStep`` and ``ColumnCleanupStep`` (always last).

    Optional steps are appended when the corresponding DataFrames are provided
    and non-empty:
    - ``AnalystEstimatesStep``   — requires ``ratings_df``.
    - ``MarketContextStep``      — requires ``sector_perf_df`` or ``industry_perf_df``.
    - ``SentimentEnrichmentStep`` — requires ``news_df`` (uses HuggingFace model).

    Args:
        indicators:       List of indicator spec dicts from ``select_alpha_indicators``.
        ratings_df:       Analyst rating history from FMP (optional).
        sector_perf_df:   Sector-level performance from FMP (optional).
        industry_perf_df: Industry-level performance from FMP (optional).
        news_df:          News articles from Alpaca (optional).

    Returns:
        A configured but not-yet-executed ``DataPipeline`` instance.
    """
    from quantrl_lab.data.processing.pipeline import DataPipeline
    from quantrl_lab.data.processing.steps import (
        AnalystEstimatesStep,
        ColumnCleanupStep,
        MarketContextStep,
        NumericConversionStep,
        SentimentEnrichmentStep,
        TechnicalIndicatorStep,
    )

    pipeline = DataPipeline().add_step(TechnicalIndicatorStep(indicators=indicators))

    if ratings_df is not None and not ratings_df.empty:
        pipeline.add_step(AnalystEstimatesStep(ratings_df=ratings_df))

    if sector_perf_df is not None or industry_perf_df is not None:
        pipeline.add_step(
            MarketContextStep(
                sector_perf_df=sector_perf_df,
                industry_perf_df=industry_perf_df,
            )
        )

    if news_df is not None and not news_df.empty:
        from quantrl_lab.data.processing.sentiment import HuggingFaceProvider

        pipeline.add_step(
            SentimentEnrichmentStep(
                news_data=news_df,
                provider=HuggingFaceProvider(),
            )
        )

    pipeline.add_step(NumericConversionStep()).add_step(ColumnCleanupStep())
    return pipeline


# ---------------------------------------------------------------------------
# Per-symbol processing
# ---------------------------------------------------------------------------


def process_symbol(
    symbol: str,
    raw_df: pd.DataFrame,
    indicators: List[Dict],
    enrichment: Optional[Dict] = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch enrichment, build pipeline, execute, and clean up for one
    symbol.

    This is the synchronous counterpart to the async enrichment flow used in
    ``multi_symbol_training.py``.  It is the right choice for single-symbol
    workflows or when concurrency is not required.

    Args:
        symbol:     Ticker symbol (e.g. ``"AAPL"``).
        raw_df:     Raw OHLCV DataFrame (index = DatetimeIndex).
        indicators: List of indicator spec dicts.
        enrichment: Optional dict with pre-fetched enrichment data::

                        {
                            "ratings_df":      pd.DataFrame | None,
                            "sector_perf_df":  pd.DataFrame | None,
                            "industry_perf_df": pd.DataFrame | None,
                            "news_df":         pd.DataFrame | None,
                        }

                    When *None*, no enrichment is applied.
        verbose:    Whether to print NaN-report and shape info.

    Returns:
        Processed DataFrame with NaN rows dropped and a ``Symbol`` column added.

    Raises:
        ValueError: If the processed DataFrame is empty after ``dropna()``.
    """
    enrichment = enrichment or {}

    pipeline = build_pipeline_for_symbol(
        indicators,
        ratings_df=enrichment.get("ratings_df"),
        sector_perf_df=enrichment.get("sector_perf_df"),
        industry_perf_df=enrichment.get("industry_perf_df"),
        news_df=enrichment.get("news_df"),
    )

    if verbose:
        console.print(f"  Pipeline for [cyan]{symbol}[/cyan]: {pipeline}")

    processed_df, _ = pipeline.execute(raw_df, symbol=symbol)

    # NaN report (visible only in verbose mode)
    if verbose:
        nan_counts = processed_df.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if not cols_with_nans.empty:
            console.print(f"  [yellow]NaN columns before dropna ({symbol}):[/yellow]")
            for col, count in cols_with_nans.items():
                console.print(f"    {col}: {count} ({count / len(processed_df) * 100:.1f}%)")

    before = len(processed_df)
    processed_df = processed_df.dropna()

    if verbose:
        console.print(
            f"  [cyan]{symbol}[/cyan]: {processed_df.shape} " f"(dropped {before - len(processed_df)} NaN rows)"
        )

    if processed_df.empty:
        raise ValueError(f"DataFrame for {symbol!r} is empty after dropna — cannot proceed.")

    processed_df["Symbol"] = symbol
    return processed_df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def train_test_split_by_date(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DatetimeIndex DataFrame into train and test sets by calendar
    date.

    The split point is determined by the *number of unique calendar dates*
    rather than the number of rows, so multi-symbol panel data with repeated
    dates is handled correctly.

    Args:
        df:           DataFrame with a ``DatetimeIndex``.
        split_ratio:  Fraction of unique dates assigned to the training set.
                      Defaults to ``0.8`` (80 % train / 20 % test).

    Returns:
        ``(train_df, test_df)`` tuple; both retain the original index.
    """
    unique_dates = df.index.unique().sort_values()
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]

    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    console.print(
        f"[cyan]Train:[/cyan] {len(train_df)} rows (until {split_date.date()})  |  "
        f"[cyan]Test:[/cyan] {len(test_df)} rows (from {split_date.date()})"
    )
    return train_df, test_df


def train_eval_test_split_by_date(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DatetimeIndex DataFrame into train, eval, and test sets by
    calendar date.

    Args:
        df:          DataFrame with a ``DatetimeIndex``.
        train_ratio: Fraction of unique dates assigned to the training set. Defaults to ``0.7``.
        eval_ratio:  Fraction of unique dates assigned to the evaluation set. Defaults to ``0.15``.
                     (The remainder goes to the test set.)

    Returns:
        ``(train_df, eval_df, test_df)`` tuple; all retain the original index.
    """
    unique_dates = df.index.unique().sort_values()

    train_split_idx = int(len(unique_dates) * train_ratio)
    eval_split_idx = int(len(unique_dates) * (train_ratio + eval_ratio))

    train_date = unique_dates[train_split_idx]
    if eval_split_idx < len(unique_dates):
        eval_date = unique_dates[eval_split_idx]
    else:
        eval_date = unique_dates[-1]

    train_df = df[df.index < train_date]
    eval_df = df[(df.index >= train_date) & (df.index < eval_date)]
    test_df = df[df.index >= eval_date]

    console.print(
        f"[cyan]Train:[/cyan] {len(train_df)} rows (until {train_date.date()})  |  "
        f"[cyan]Eval:[/cyan] {len(eval_df)} rows (until {eval_date.date()})  |  "
        f"[cyan]Test:[/cyan] {len(test_df)} rows (from {eval_date.date()})"
    )
    return train_df, eval_df, test_df

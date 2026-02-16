import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class AlphaJob:
    """Defines a single alpha research task."""

    data: pd.DataFrame  # The data to test on
    indicator_name: str  # e.g., "RSI" (for calculation)
    strategy_name: str  # e.g., "mean_reversion" (for logic)
    indicator_params: Dict[str, Any] = field(default_factory=dict)  # e.g. {"window": 14}
    strategy_params: Dict[str, Any] = field(default_factory=dict)  # e.g. {"oversold": 30}
    allow_short: bool = True

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SignalMetrics:
    """Statistical metrics for evaluating the predictive power of a
    signal."""

    ic: float  # Information Coefficient (Spearman Rank Correlation)
    rank_ic: float
    ic_p_value: float
    information_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    turnover: float  # Signal stability


@dataclass
class AlphaResult:
    """Results of an alpha research job."""

    job: AlphaJob
    metrics: Dict[str, Any]  # Use Any for Dict[str, float] and SignalMetrics
    equity_curve: Optional[pd.Series] = None
    signals: Optional[pd.Series] = None
    status: str = "completed"
    error: Optional[Exception] = None

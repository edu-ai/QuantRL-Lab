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
class AlphaResult:
    """Results of an alpha research job."""

    job: AlphaJob
    metrics: Dict[str, Any]
    equity_curve: Optional[pd.Series] = None
    signals: Optional[pd.Series] = None
    status: str = "completed"
    error: Optional[str] = None

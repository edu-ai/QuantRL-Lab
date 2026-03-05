"""
Processing steps for data pipeline.

This package provides composable transformation steps that can be
chained together in a DataPipeline. Each step implements the
ProcessingStep protocol.
"""

from .alternative.analyst import AnalystEstimatesStep
from .alternative.sentiment import SentimentEnrichmentStep
from .base import ProcessingStep
from .cleaning.cleanup import ColumnCleanupStep
from .cleaning.conversion import NumericConversionStep
from .features.context import MarketContextStep
from .features.cross_sectional import CrossSectionalStep
from .features.technical import TechnicalIndicatorStep

__all__ = [
    "ProcessingStep",
    "TechnicalIndicatorStep",
    "CrossSectionalStep",
    "SentimentEnrichmentStep",
    "ColumnCleanupStep",
    "NumericConversionStep",
    "AnalystEstimatesStep",
    "MarketContextStep",
]

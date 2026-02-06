"""
Feature engineering module for cryptocurrency prediction.

This module provides:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Lag and rolling features
- Temporal features
- Feature pipeline orchestration
"""

from .technical import TechnicalIndicators
from .lag_features import LagFeatureGenerator
from .pipeline import FeaturePipeline

__all__ = [
    "TechnicalIndicators",
    "LagFeatureGenerator",
    "FeaturePipeline"
]

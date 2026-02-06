"""
Evaluation module for model performance assessment.

This module provides:
- Comprehensive metrics calculation
- Baseline model comparison
- Backtesting utilities
- Visualization tools
"""

from .metrics import calculate_metrics, MetricsCalculator
from .backtester import Backtester
from .visualizer import PredictionVisualizer

__all__ = [
    "calculate_metrics",
    "MetricsCalculator",
    "Backtester",
    "PredictionVisualizer"
]

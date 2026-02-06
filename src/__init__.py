"""
Crypto Forecast - Production-Ready Cryptocurrency Price Prediction Pipeline

This package provides a modular, scalable architecture for:
- Fetching cryptocurrency market data from CoinGecko/CoinMarketCap
- Engineering predictive features (technical indicators, lag features, etc.)
- Training XGBoost/LightGBM models with time-series cross-validation
- Generating multi-horizon forecasts for price and market cap
- Evaluating model performance with comprehensive metrics

Author: Refactored Architecture
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Crypto Forecast Team"

from .config import Config
from .pipeline.orchestrator import Pipeline

__all__ = ["Config", "Pipeline", "__version__"]

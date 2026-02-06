"""
Data module for cryptocurrency data ingestion and management.

This module provides:
- CoinGecko API client for fetching market data
- Data loading and saving utilities
- Data validation and cleaning
"""

from .ingestion import CoinGeckoClient, DataIngestion
from .loader import DataLoader
from .validator import DataValidator

__all__ = [
    "CoinGeckoClient",
    "DataIngestion", 
    "DataLoader",
    "DataValidator"
]

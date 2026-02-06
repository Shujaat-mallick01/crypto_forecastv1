"""
Data loader module for loading and managing processed data.

Provides utilities for:
- Loading raw and processed data
- Combining data from multiple coins
- Data caching and lazy loading
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading and management utilities.
    
    Handles loading raw and processed data files with caching
    and validation.
    """
    
    def __init__(self, config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_raw_data(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load raw data for a symbol.
        
        Args:
            symbol: Coin symbol
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame or None
        """
        cache_key = f"raw_{symbol}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Find latest file
        pattern = f"historical_{symbol}_*.csv"
        files = list(self.config.raw_data_path.glob(pattern))
        
        if not files:
            logger.warning(f"No raw data found for {symbol}")
            return None
        
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file, parse_dates=['timestamp'])
            
            if use_cache:
                self._cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {latest_file}: {e}")
            return None
    
    def load_processed_data(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load processed (feature-engineered) data for a symbol.
        
        Args:
            symbol: Coin symbol
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame or None
        """
        cache_key = f"processed_{symbol}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        filepath = self.config.processed_data_path / f"{symbol}_features.csv"
        
        if not filepath.exists():
            logger.warning(f"Processed data not found for {symbol}")
            return None
        
        try:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            
            if use_cache:
                self._cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_raw_data(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load raw data for multiple symbols.
        
        Args:
            symbols: List of symbols (default: all configured)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        symbols = symbols or self.config.coin_symbols
        results = {}
        
        for symbol in symbols:
            df = self.load_raw_data(symbol)
            if df is not None:
                results[symbol] = df
        
        return results
    
    def load_all_processed_data(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load processed data for multiple symbols.
        
        Args:
            symbols: List of symbols (default: all configured)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        symbols = symbols or self.config.coin_symbols
        results = {}
        
        for symbol in symbols:
            df = self.load_processed_data(symbol)
            if df is not None:
                results[symbol] = df
        
        return results
    
    def combine_data(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine data from multiple coins into single DataFrame.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            
        Returns:
            Combined DataFrame with 'symbol' column
        """
        if not data_dict:
            return pd.DataFrame()
        
        dfs = []
        for symbol, df in data_dict.items():
            df_copy = df.copy()
            if 'symbol' not in df_copy.columns:
                df_copy['symbol'] = symbol
            dfs.append(df_copy)
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(['symbol', 'timestamp'])
        
        return combined
    
    def get_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train and test sets.
        
        Args:
            df: Input DataFrame (must be sorted by timestamp)
            test_size: Fraction for test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.sort_values('timestamp')
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        return train_df, test_df
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Path:
        """
        Save processed data to file.
        
        Args:
            df: Processed DataFrame
            symbol: Coin symbol
            
        Returns:
            Path to saved file
        """
        self.config.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        filepath = self.config.processed_data_path / f"{symbol}_features.csv"
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved processed data to {filepath}")
        
        return filepath
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear data cache.
        
        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            keys_to_remove = [k for k in self._cache if symbol in k]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
        
        logger.debug("Cache cleared")
    
    def get_available_data(self) -> Dict[str, Dict]:
        """
        Get information about available data files.
        
        Returns:
            Dictionary with data availability info
        """
        info = {
            'raw': {},
            'processed': {}
        }
        
        # Check raw data
        for symbol in self.config.coin_symbols:
            pattern = f"historical_{symbol}_*.csv"
            files = list(self.config.raw_data_path.glob(pattern))
            
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                info['raw'][symbol] = {
                    'file': str(latest),
                    'modified': datetime.fromtimestamp(latest.stat().st_mtime)
                }
        
        # Check processed data
        for symbol in self.config.coin_symbols:
            filepath = self.config.processed_data_path / f"{symbol}_features.csv"
            
            if filepath.exists():
                info['processed'][symbol] = {
                    'file': str(filepath),
                    'modified': datetime.fromtimestamp(filepath.stat().st_mtime)
                }
        
        return info

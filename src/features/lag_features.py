"""
Lag and rolling feature generation module.

Provides features based on:
- Lagged values (previous time steps)
- Rolling statistics (mean, std, min, max)
- Return calculations
- Temporal encoding
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LagFeatureGenerator:
    """
    Generate lag-based and rolling features for time series data.
    
    Features include:
    - Lagged values of key columns
    - Rolling statistics (mean, std, min, max)
    - Percentage returns over various periods
    - Temporal features (day of week, month, etc.)
    """
    
    def __init__(
        self,
        lag_periods: List[int] = [1, 3, 7, 14, 30],
        rolling_windows: List[int] = [7, 14, 30, 60],
        return_periods: List[int] = [1, 3, 7, 14, 30]
    ):
        """
        Initialize lag feature generator.
        
        Args:
            lag_periods: Periods for lagged features
            rolling_windows: Windows for rolling statistics
            return_periods: Periods for return calculations
        """
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.return_periods = return_periods
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['price', 'volume', 'market_cap']
    ) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for period in self.lag_periods:
                df[f'{col}_lag_{period}'] = df[col].shift(period)
        
        logger.debug(f"Created lag features for periods: {self.lag_periods}")
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['price', 'volume', 'market_cap']
    ) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate rolling stats for
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for window in self.rolling_windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = (
                    df[col].rolling(window=window).mean()
                )
                
                # Rolling standard deviation
                df[f'{col}_rolling_std_{window}'] = (
                    df[col].rolling(window=window).std()
                )
                
                # Rolling min/max
                df[f'{col}_rolling_min_{window}'] = (
                    df[col].rolling(window=window).min()
                )
                df[f'{col}_rolling_max_{window}'] = (
                    df[col].rolling(window=window).max()
                )
                
                # Rolling range (max - min)
                df[f'{col}_rolling_range_{window}'] = (
                    df[f'{col}_rolling_max_{window}'] - 
                    df[f'{col}_rolling_min_{window}']
                )
        
        logger.debug(f"Created rolling features for windows: {self.rolling_windows}")
        
        return df
    
    def create_return_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['price', 'market_cap']
    ) -> pd.DataFrame:
        """
        Calculate percentage returns over different periods.
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate returns for
            
        Returns:
            DataFrame with return features added
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for period in self.return_periods:
                # Percentage return
                df[f'{col}_return_{period}d'] = (
                    (df[col] - df[col].shift(period)) / 
                    df[col].shift(period) * 100
                )
                
                # Log return (more stable for ML)
                df[f'{col}_log_return_{period}d'] = (
                    np.log(df[col] / df[col].shift(period))
                )
        
        logger.debug(f"Created return features for periods: {self.return_periods}")
        
        return df
    
    def create_volatility_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'price'
    ) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            
        Returns:
            DataFrame with volatility features added
        """
        df = df.copy()
        
        if price_col not in df.columns:
            return df
        
        # Daily returns
        df['daily_return'] = df[price_col].pct_change() * 100
        
        for window in self.rolling_windows:
            # Volatility (rolling std of returns)
            df[f'volatility_{window}d'] = (
                df['daily_return'].rolling(window=window).std()
            )
            
            # Realized volatility (annualized)
            df[f'realized_vol_{window}d'] = (
                df['daily_return'].rolling(window=window).std() * np.sqrt(365)
            )
        
        logger.debug("Created volatility features")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal/calendar features from timestamp.
        
        Args:
            df: Input DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found for temporal features")
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
        
        # Cyclical encoding (better for ML models)
        # Day of week (0-6)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of month (1-31)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month start/end indicators
        df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
        
        logger.debug("Created temporal features")
        
        return df
    
    def create_ratio_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        volume_col: str = 'volume',
        market_cap_col: str = 'market_cap'
    ) -> pd.DataFrame:
        """
        Create ratio-based features.
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            volume_col: Volume column name
            market_cap_col: Market cap column name
            
        Returns:
            DataFrame with ratio features added
        """
        df = df.copy()
        
        # Price to moving average ratios
        for window in self.rolling_windows:
            ma_col = f'{price_col}_rolling_mean_{window}'
            if ma_col in df.columns:
                df[f'price_to_ma_{window}_ratio'] = df[price_col] / df[ma_col]
        
        # Volume ratios
        if volume_col in df.columns:
            for window in self.rolling_windows:
                vol_ma = df[volume_col].rolling(window=window).mean()
                df[f'volume_ratio_{window}d'] = df[volume_col] / vol_ma
        
        # Market cap to volume ratio
        if volume_col in df.columns and market_cap_col in df.columns:
            df['mcap_to_volume_ratio'] = df[market_cap_col] / df[volume_col].replace(0, np.nan)
        
        logger.debug("Created ratio features")
        
        return df
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['price', 'volume', 'market_cap']
    ) -> pd.DataFrame:
        """
        Create all lag and rolling features.
        
        Args:
            df: Input DataFrame
            columns: Columns to generate features for
            
        Returns:
            DataFrame with all features added
        """
        logger.info("Creating lag and rolling features...")
        
        # Lag features
        df = self.create_lag_features(df, columns)
        
        # Rolling features
        df = self.create_rolling_features(df, columns)
        
        # Return features
        df = self.create_return_features(df, ['price', 'market_cap'])
        
        # Volatility features
        df = self.create_volatility_features(df)
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Ratio features
        df = self.create_ratio_features(df)
        
        logger.info(f"Total columns after feature engineering: {len(df.columns)}")
        
        return df

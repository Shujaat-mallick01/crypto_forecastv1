"""
Technical indicators module for cryptocurrency analysis.

Provides implementations of common technical analysis indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- ATR (Average True Range)
- Momentum indicators
"""

import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical analysis indicators for price data.
    
    All methods are designed to work with pandas DataFrames
    and return the DataFrame with new indicator columns added.
    """
    
    @staticmethod
    def sma(
        df: pd.DataFrame,
        column: str = 'close',
        periods: list = [7, 14, 30, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate SMA on
            periods: List of periods for SMA
            
        Returns:
            DataFrame with SMA columns added
        """
        df = df.copy()
        
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def ema(
        df: pd.DataFrame,
        column: str = 'close',
        periods: list = [12, 26, 50]
    ) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate EMA on
            periods: List of periods for EMA
            
        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        
        for period in periods:
            df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def rsi(
        df: pd.DataFrame,
        column: str = 'close',
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        RSI measures the speed and magnitude of recent price changes
        to evaluate overvalued or undervalued conditions.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate RSI on
            period: RSI period (typically 14)
            
        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()
        
        # Calculate price changes
        delta = df[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.inf)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        df[f'rsi_{period}'] = df[f'rsi_{period}'].fillna(50)
        df[f'rsi_{period}'] = df[f'rsi_{period}'].clip(0, 100)
        
        return df
    
    @staticmethod
    def macd(
        df: pd.DataFrame,
        column: str = 'close',
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD shows the relationship between two moving averages
        and is used to identify trend changes.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate MACD on
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD, signal, and histogram columns
        """
        df = df.copy()
        
        # Calculate EMAs
        ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        column: str = 'close',
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands show price volatility with upper and lower
        bands based on standard deviation from a moving average.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate bands on
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Band columns
        """
        df = df.copy()
        
        # Middle band (SMA)
        df['bb_middle'] = df[column].rolling(window=period).mean()
        
        # Standard deviation
        rolling_std = df[column].rolling(window=period).std()
        
        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Band width (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # %B (position within bands)
        df['bb_percent'] = (
            (df[column] - df['bb_lower']) / 
            (df['bb_upper'] - df['bb_lower'])
        )
        
        return df
    
    @staticmethod
    def atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        ATR measures market volatility by decomposing the entire
        range of an asset price for a period.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            DataFrame with ATR column
        """
        df = df.copy()
        
        # True Range components
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        # True Range is the max of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR is the smoothed average of True Range
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def momentum(
        df: pd.DataFrame,
        column: str = 'close',
        periods: list = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        Calculate price momentum (rate of change).
        
        Args:
            df: DataFrame with price data
            column: Column to calculate momentum on
            periods: List of periods
            
        Returns:
            DataFrame with momentum columns
        """
        df = df.copy()
        
        for period in periods:
            df[f'momentum_{period}'] = (
                (df[column] - df[column].shift(period)) / 
                df[column].shift(period)
            ) * 100
        
        return df
    
    @staticmethod
    def stochastic_oscillator(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Shows the location of the close relative to the high-low
        range over a set number of periods.
        
        Args:
            df: DataFrame with OHLC data
            k_period: %K period
            d_period: %D smoothing period
            
        Returns:
            DataFrame with %K and %D columns
        """
        df = df.copy()
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = (
            (df['close'] - lowest_low) / 
            (highest_high - lowest_low)
        ) * 100
        
        # Calculate %D (smoothed %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume.
        
        OBV uses volume flow to predict changes in price.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with OBV column
        """
        df = df.copy()
        
        # Direction of price change
        price_change = df['close'].diff()
        
        # OBV calculation
        obv = []
        prev_obv = 0
        
        for i, row in df.iterrows():
            if i == df.index[0]:
                obv.append(row['volume'])
            elif price_change.loc[i] > 0:
                prev_obv += row['volume']
                obv.append(prev_obv)
            elif price_change.loc[i] < 0:
                prev_obv -= row['volume']
                obv.append(prev_obv)
            else:
                obv.append(prev_obv)
        
        df['obv'] = obv
        
        return df
    
    @classmethod
    def add_all_indicators(
        cls,
        df: pd.DataFrame,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLC and volume data
            rsi_period: RSI calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviations
            
        Returns:
            DataFrame with all indicators added
        """
        logger.info("Adding technical indicators...")
        
        # Moving averages
        df = cls.sma(df, periods=[7, 14, 30, 50])
        df = cls.ema(df, periods=[12, 26])
        
        # Momentum indicators
        df = cls.rsi(df, period=rsi_period)
        df = cls.macd(df, fast_period=macd_fast, slow_period=macd_slow, 
                      signal_period=macd_signal)
        df = cls.momentum(df, periods=[7, 14, 30])
        
        # Volatility indicators
        df = cls.bollinger_bands(df, period=bb_period, std_dev=bb_std)
        df = cls.atr(df, period=14)
        
        # Volume indicators (if volume available)
        if 'volume' in df.columns and df['volume'].notna().any():
            df = cls.obv(df)
        
        # Stochastic (if OHLC available)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df = cls.stochastic_oscillator(df)
        
        logger.info(f"Added {len(df.columns)} indicator columns")
        
        return df

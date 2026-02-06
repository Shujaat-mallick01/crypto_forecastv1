"""
Feature engineering pipeline orchestration.

Combines all feature generation steps into a unified pipeline
with support for multi-target prediction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from .technical import TechnicalIndicators
from .lag_features import LagFeatureGenerator

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline.
    
    Orchestrates:
    - Technical indicator calculation
    - Lag and rolling feature generation
    - Target variable creation
    - Feature scaling
    - Feature selection
    """
    
    # Columns to exclude from features (metadata and targets)
    EXCLUDE_COLUMNS = [
        'timestamp', 'symbol', 'coin_id', 'name',
        'daily_return'  # Intermediate calculation
    ]
    
    def __init__(self, config):
        """
        Initialize feature pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_config = config.feature_config
        self.model_config = config.model_config
        
        # Initialize feature generators
        self.lag_generator = LagFeatureGenerator(
            lag_periods=self.feature_config.lag_periods,
            rolling_windows=self.feature_config.rolling_windows,
            return_periods=self.feature_config.return_periods
        )
        
        # Scalers (fitted during transform)
        self.scalers: Dict[str, object] = {}
        self.feature_names: List[str] = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw data.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Creating features for {len(df)} rows...")
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').copy()
        
        # Step 1: Technical indicators
        df = TechnicalIndicators.add_all_indicators(
            df,
            rsi_period=self.feature_config.rsi_period,
            macd_fast=self.feature_config.macd_fast,
            macd_slow=self.feature_config.macd_slow,
            macd_signal=self.feature_config.macd_signal,
            bb_period=self.feature_config.bollinger_period,
            bb_std=self.feature_config.bollinger_std
        )
        
        # Step 2: Lag and rolling features
        df = self.lag_generator.create_all_features(df)
        
        # Step 3: Create target variables
        df = self._create_targets(df)
        
        # Step 4: Handle outliers (optional)
        if self.feature_config.clip_outliers:
            df = self._clip_outliers(df)
        
        # Step 5: Remove NaN rows
        initial_rows = len(df)
        df = df.dropna()
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with NaN values ({removed/initial_rows*100:.1f}%)")
        
        logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for all horizons.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target columns added
        """
        for target in self.model_config.targets:
            col = target if target in df.columns else 'price' if target == 'price' else 'market_cap'
            
            if col not in df.columns:
                logger.warning(f"Target column {col} not found")
                continue
            
            for horizon in self.model_config.horizons:
                target_col = f'target_{target}_{horizon}d'
                df[target_col] = df[col].shift(-horizon)
        
        logger.debug(f"Created targets for horizons: {self.model_config.horizons}")
        
        return df
    
    def _clip_outliers(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Clip extreme outliers based on z-score.
        
        Args:
            df: DataFrame with features
            threshold: Z-score threshold (default from config)
            
        Returns:
            DataFrame with outliers clipped
        """
        threshold = threshold or self.feature_config.outlier_threshold
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['day_of_week', 'month', 'day_of_month', 'quarter', 'is_weekend']
        
        for col in numeric_cols:
            if col in exclude or col.startswith('target_'):
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                lower = mean - threshold * std
                upper = mean + threshold * std
                df[col] = df[col].clip(lower, upper)
        
        logger.debug(f"Clipped outliers with z-threshold: {threshold}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude = self.EXCLUDE_COLUMNS.copy()
        
        # Also exclude raw price columns and targets
        exclude.extend(['open', 'high', 'low', 'close', 'price', 
                       'volume', 'volume_24h', 'market_cap'])
        
        # Exclude target columns
        for target in self.model_config.targets:
            for horizon in self.model_config.horizons:
                exclude.append(f'target_{target}_{horizon}d')
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude and not col.startswith('target_')
        ]
        
        self.feature_names = feature_cols
        
        return feature_cols
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target: str = 'price',
        horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for model training.
        
        Args:
            df: Featured DataFrame
            target: Target type ('price' or 'market_cap')
            horizon: Forecast horizon in days
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        target_col = f'target_{target}_{horizon}d'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Get feature columns
        feature_cols = self.get_feature_names(df)
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with NaN target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Prepared data for {target} @ {horizon}d: {len(X)} samples, {len(feature_cols)} features")
        
        return X, y, feature_cols
    
    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True,
        scaler_key: str = 'default'
    ) -> pd.DataFrame:
        """
        Scale features using configured method.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training)
            scaler_key: Key to store/retrieve scaler
            
        Returns:
            Scaled DataFrame
        """
        method = self.feature_config.scaling_method
        
        if fit:
            if method == 'robust':
                scaler = RobustScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaling method: {method}, using robust")
                scaler = RobustScaler()
            
            scaled_data = scaler.fit_transform(X)
            self.scalers[scaler_key] = scaler
        else:
            if scaler_key not in self.scalers:
                raise ValueError(f"Scaler '{scaler_key}' not fitted")
            
            scaled_data = self.scalers[scaler_key].transform(X)
        
        return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
    
    def get_feature_importance_columns(self) -> Dict[str, List[str]]:
        """
        Get feature columns grouped by type.
        
        Returns:
            Dictionary mapping feature type to column names
        """
        groups = {
            'technical': [],
            'lag': [],
            'rolling': [],
            'return': [],
            'volatility': [],
            'temporal': [],
            'ratio': []
        }
        
        for col in self.feature_names:
            if any(x in col for x in ['rsi', 'macd', 'bb_', 'sma', 'ema', 'atr', 'stoch', 'obv']):
                groups['technical'].append(col)
            elif '_lag_' in col:
                groups['lag'].append(col)
            elif '_rolling_' in col:
                groups['rolling'].append(col)
            elif '_return_' in col or 'log_return' in col:
                groups['return'].append(col)
            elif 'volatility' in col or 'vol_' in col:
                groups['volatility'].append(col)
            elif any(x in col for x in ['day_of', 'month', 'quarter', 'week', 'weekend']):
                groups['temporal'].append(col)
            elif 'ratio' in col:
                groups['ratio'].append(col)
        
        return groups


def process_coin(
    symbol: str,
    df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Process features for a single coin.
    
    Args:
        symbol: Coin symbol
        df: Raw data DataFrame
        config: Configuration object
        
    Returns:
        Featured DataFrame
    """
    logger.info(f"Processing features for {symbol}...")
    
    pipeline = FeaturePipeline(config)
    featured_df = pipeline.create_features(df)
    
    # Ensure symbol column
    if 'symbol' not in featured_df.columns:
        featured_df['symbol'] = symbol
    
    return featured_df

"""
Metrics calculation module for model evaluation.

Provides comprehensive metrics including:
- Regression metrics (MAE, RMSE, MAPE, R²)
- Directional accuracy
- Custom financial metrics
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan}
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (handling zeros)
    mask_nonzero = y_true != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
    else:
        mape = np.nan
    
    # R² score
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


class MetricsCalculator:
    """
    Comprehensive metrics calculator with additional financial metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics_history: List[Dict] = []
    
    def calculate(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        prices: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            prices: Price data for directional accuracy
            
        Returns:
            Dictionary of all metrics
        """
        # Basic metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Directional accuracy
        metrics['directional_accuracy'] = self.directional_accuracy(y_true, y_pred)
        
        # Additional metrics
        metrics['max_error'] = self.max_error(y_true, y_pred)
        metrics['median_ae'] = self.median_absolute_error(y_true, y_pred)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    @staticmethod
    def directional_accuracy(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate directional accuracy (trend prediction).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Percentage of correct direction predictions
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) < 2:
            return np.nan
        
        # Calculate changes
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct = (true_direction == pred_direction).sum()
        total = len(true_direction)
        
        return (correct / total) * 100 if total > 0 else np.nan
    
    @staticmethod
    def max_error(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """Calculate maximum absolute error."""
        return np.max(np.abs(np.array(y_true) - np.array(y_pred)))
    
    @staticmethod
    def median_absolute_error(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """Calculate median absolute error."""
        return np.median(np.abs(np.array(y_true) - np.array(y_pred)))
    
    @staticmethod
    def symmetric_mape(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate Symmetric MAPE (handles zeros better).
        
        sMAPE = 100 * mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if mask.sum() == 0:
            return np.nan
        
        return 100 * np.mean(numerator[mask] / denominator[mask])
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all metrics history."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple models.
        
        Args:
            results: Dictionary mapping model names to metrics
            
        Returns:
            DataFrame comparison
        """
        comparison = pd.DataFrame(results).T
        comparison.index.name = 'Model'
        
        # Add rankings
        for col in comparison.columns:
            if col in ['r2', 'directional_accuracy']:
                comparison[f'{col}_rank'] = comparison[col].rank(ascending=False)
            else:
                comparison[f'{col}_rank'] = comparison[col].rank(ascending=True)
        
        return comparison


class BaselineMetrics:
    """
    Baseline model implementations for comparison.
    """
    
    @staticmethod
    def naive_baseline(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Naive baseline: predict tomorrow = today.
        
        Args:
            y: Time series data
            
        Returns:
            Predictions (shifted by 1)
        """
        y = np.array(y)
        return np.concatenate([[y[0]], y[:-1]])
    
    @staticmethod
    def moving_average_baseline(
        y: Union[pd.Series, np.ndarray],
        window: int = 7
    ) -> np.ndarray:
        """
        Moving average baseline.
        
        Args:
            y: Time series data
            window: Moving average window
            
        Returns:
            MA predictions
        """
        series = pd.Series(y)
        ma = series.rolling(window=window).mean()
        ma = ma.fillna(series.expanding().mean())
        return ma.values
    
    @staticmethod
    def drift_baseline(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Drift baseline: linear extrapolation.
        
        Args:
            y: Time series data
            
        Returns:
            Linear trend predictions
        """
        y = np.array(y)
        n = len(y)
        
        if n < 2:
            return y
        
        # Calculate overall drift
        drift = (y[-1] - y[0]) / (n - 1)
        
        # Predictions
        predictions = np.zeros(n)
        predictions[0] = y[0]
        
        for i in range(1, n):
            predictions[i] = y[i-1] + drift
        
        return predictions
    
    def evaluate_baselines(
        self,
        y_true: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all baseline methods.
        
        Args:
            y_true: True target values
            
        Returns:
            Dictionary of metrics for each baseline
        """
        y_true = np.array(y_true)
        
        baselines = {
            'naive': self.naive_baseline(y_true),
            'ma_7': self.moving_average_baseline(y_true, 7),
            'ma_30': self.moving_average_baseline(y_true, 30),
            'drift': self.drift_baseline(y_true)
        }
        
        results = {}
        for name, y_pred in baselines.items():
            results[name] = calculate_metrics(y_true, y_pred)
        
        return results

"""
Backtesting module for walk-forward validation.

Provides time-series specific backtesting with:
- Walk-forward validation
- Expanding window training
- Rolling window training
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np

from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    Walk-forward backtesting for time series models.
    
    Supports:
    - Expanding window (use all available history)
    - Rolling window (fixed-size training window)
    - Step-forward validation
    """
    
    def __init__(
        self,
        train_size: int = 252,  # ~1 year of daily data
        test_size: int = 30,    # ~1 month
        step_size: int = 7,     # Weekly retraining
        method: str = 'expanding'  # 'expanding' or 'rolling'
    ):
        """
        Initialize backtester.
        
        Args:
            train_size: Minimum training window size
            test_size: Test/validation window size
            step_size: Step forward size for each iteration
            method: 'expanding' or 'rolling' window
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.method = method
        
        self.results: List[Dict] = []
    
    def backtest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run walk-forward backtest.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_factory: Callable that returns a new model instance
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with backtest results
        """
        n_samples = len(X)
        self.results = []
        
        # Calculate number of iterations
        n_iterations = (n_samples - self.train_size - self.test_size) // self.step_size + 1
        
        logger.info(f"Starting backtest: {n_iterations} iterations")
        logger.info(f"Method: {self.method}, Train: {self.train_size}, Test: {self.test_size}, Step: {self.step_size}")
        
        predictions_all = []
        actuals_all = []
        
        for i in range(n_iterations):
            # Calculate window boundaries
            if self.method == 'expanding':
                train_start = 0
            else:  # rolling
                train_start = i * self.step_size
            
            train_end = self.train_size + i * self.step_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train model
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            # Store results
            self.results.append({
                'iteration': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': train_end - train_start,
                **metrics
            })
            
            predictions_all.extend(y_pred)
            actuals_all.extend(y_test.values)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Iteration {i + 1}/{n_iterations}: RMSE={metrics['rmse']:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Add overall metrics
        overall_metrics = calculate_metrics(
            np.array(actuals_all),
            np.array(predictions_all)
        )
        
        logger.info(f"Backtest complete. Overall RMSE: {overall_metrics['rmse']:.4f}")
        
        return results_df
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics from backtest.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        summary = {
            'n_iterations': len(results_df),
            'mae_mean': results_df['mae'].mean(),
            'mae_std': results_df['mae'].std(),
            'rmse_mean': results_df['rmse'].mean(),
            'rmse_std': results_df['rmse'].std(),
            'mape_mean': results_df['mape'].mean(),
            'mape_std': results_df['mape'].std(),
            'r2_mean': results_df['r2'].mean(),
            'r2_std': results_df['r2'].std()
        }
        
        return summary
    
    def analyze_stability(self) -> Dict[str, float]:
        """
        Analyze model stability over time.
        
        Returns:
            Dictionary of stability metrics
        """
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        # Check for performance degradation
        half = len(results_df) // 2
        first_half_rmse = results_df['rmse'].iloc[:half].mean()
        second_half_rmse = results_df['rmse'].iloc[half:].mean()
        
        stability = {
            'first_half_rmse': first_half_rmse,
            'second_half_rmse': second_half_rmse,
            'performance_change': (second_half_rmse - first_half_rmse) / first_half_rmse * 100,
            'rmse_coefficient_of_variation': results_df['rmse'].std() / results_df['rmse'].mean() * 100,
            'worst_rmse': results_df['rmse'].max(),
            'best_rmse': results_df['rmse'].min()
        }
        
        return stability


class WalkForwardOptimizer:
    """
    Walk-forward optimization with periodic model retraining.
    """
    
    def __init__(
        self,
        retrain_frequency: int = 30,  # Retrain every N days
        min_train_size: int = 180,    # Minimum training data
        validation_size: int = 30     # Validation for hyperparameter tuning
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            retrain_frequency: Days between model retraining
            min_train_size: Minimum training samples required
            validation_size: Validation set size for tuning
        """
        self.retrain_frequency = retrain_frequency
        self.min_train_size = min_train_size
        self.validation_size = validation_size
        
        self.training_history: List[Dict] = []
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        optimizer: Optional[Callable] = None
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Run walk-forward optimization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_factory: Callable returning model instance
            optimizer: Optional hyperparameter optimizer
            
        Returns:
            Tuple of (predictions DataFrame, training history)
        """
        n_samples = len(X)
        
        all_predictions = []
        self.training_history = []
        
        current_model = None
        last_train_idx = 0
        
        for i in range(self.min_train_size, n_samples):
            # Check if retraining is needed
            days_since_training = i - last_train_idx
            
            if current_model is None or days_since_training >= self.retrain_frequency:
                # Retrain model
                train_end = i - self.validation_size
                val_end = i
                
                X_train = X.iloc[:train_end]
                y_train = y.iloc[:train_end]
                X_val = X.iloc[train_end:val_end]
                y_val = y.iloc[train_end:val_end]
                
                # Optionally optimize hyperparameters
                if optimizer is not None:
                    best_params = optimizer(X_train, y_train, X_val, y_val)
                    current_model = model_factory(**best_params)
                else:
                    current_model = model_factory()
                
                current_model.fit(X_train, y_train, X_val, y_val)
                
                last_train_idx = i
                
                self.training_history.append({
                    'train_idx': i,
                    'train_samples': len(X_train),
                    'timestamp': datetime.now()
                })
                
                logger.debug(f"Retrained model at index {i}")
            
            # Make prediction
            X_current = X.iloc[[i]]
            pred = current_model.predict(X_current)[0]
            
            all_predictions.append({
                'index': i,
                'actual': y.iloc[i],
                'predicted': pred,
                'model_train_idx': last_train_idx
            })
        
        return pd.DataFrame(all_predictions), self.training_history

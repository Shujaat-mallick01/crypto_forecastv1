"""
LightGBM model implementation for cryptocurrency prediction.

Provides an optimized LightGBM regressor with:
- Faster training than XGBoost
- Early stopping support
- Feature importance tracking
- Memory efficiency
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import lightgbm as lgb

from .base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM regression model for price/market cap prediction.
    
    Features:
    - Gradient boosting with leaf-wise tree growth
    - Faster training than XGBoost (typically 2-3x)
    - Lower memory usage
    - Built-in categorical feature support
    """
    
    DEFAULT_PARAMS = {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
        'early_stopping_rounds': 50
    }
    
    def __init__(
        self,
        name: str = 'lightgbm',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LightGBM model.
        
        Args:
            name: Model name
            params: Model hyperparameters (merged with defaults)
        """
        # Merge with defaults
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)
        
        super().__init__(name, merged_params)
        
        # Extract early stopping (not a model param)
        self.early_stopping_rounds = merged_params.pop('early_stopping_rounds', 50)
        
        # Initialize model
        self.model = lgb.LGBMRegressor(**merged_params)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'LightGBMModel':
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets (for early stopping)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training LightGBM model on {len(X_train)} samples...")
        
        self.feature_names = list(X_train.columns)
        
        # Prepare evaluation set for early stopping
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Prepare callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=False
            ),
            lgb.log_evaluation(period=0)  # Suppress logging
        ]
        
        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        self.is_fitted = True
        self.metadata['n_samples'] = len(X_train)
        self.metadata['n_features'] = len(self.feature_names)
        self.metadata['best_iteration'] = self.model.best_iteration_
        
        logger.info(f"LightGBM training complete. Best iteration: {self.model.best_iteration_}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def set_params(self, **params) -> 'LightGBMModel':
        """Set model parameters."""
        self.model.set_params(**params)
        self.params.update(params)
        return self


class LightGBMModelWithOptuna(LightGBMModel):
    """
    LightGBM model with Optuna hyperparameter optimization.
    """
    
    def __init__(
        self,
        name: str = 'lightgbm_optuna',
        params: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        timeout: int = 3600
    ):
        """
        Initialize LightGBM model with Optuna support.
        
        Args:
            name: Model name
            params: Base hyperparameters
            n_trials: Number of Optuna trials
            timeout: Optimization timeout in seconds
        """
        super().__init__(name, params)
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of best parameters
        """
        try:
            import optuna
            from sklearn.metrics import mean_squared_error
        except ImportError:
            logger.warning("Optuna not installed. Using default parameters.")
            return self.params
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'n_jobs': -1,
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            
            return rmse
        
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials...")
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        
        # Update model with best parameters
        self.params.update(self.best_params)
        self.model = lgb.LGBMRegressor(**self.params, verbose=-1)
        
        return self.best_params

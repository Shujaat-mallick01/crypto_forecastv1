"""
XGBoost model implementation for cryptocurrency prediction.

Provides an optimized XGBoost regressor with:
- Early stopping support
- GPU acceleration (optional)
- Feature importance tracking
- Hyperparameter configuration
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import xgboost as xgb

from .base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost regression model for price/market cap prediction.
    
    Features:
    - Gradient boosted trees optimized for tabular data
    - Early stopping to prevent overfitting
    - Built-in feature importance
    - GPU support when available
    """
    
    DEFAULT_PARAMS = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 50
    }
    
    def __init__(
        self,
        name: str = 'xgboost',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize XGBoost model.
        
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
        self.model = xgb.XGBRegressor(**merged_params)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets (for early stopping)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples...")
        
        self.feature_names = list(X_train.columns)
        
        # Prepare evaluation set for early stopping
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        self.is_fitted = True
        self.metadata['n_samples'] = len(X_train)
        self.metadata['n_features'] = len(self.feature_names)
        # self.metadata['best_iteration'] = self.model.best_iteration
        
        try: 
            self.metadata['best_iteration'] = self.model.best_iteration
            logger.info(f"XGBoost training complete. Best iteration: {self.model.best_iteration}")
        except AttributeError:
            self.metadata['best_iteration'] = self.model.n_estimators
            logger.info(f"XGBoost training complete. Used {self.model.n_estimators} estimators (no early stopping)")
                
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
    
    def set_params(self, **params) -> 'XGBoostModel':
        """Set model parameters."""
        self.model.set_params(**params)
        self.params.update(params)
        return self


class XGBoostModelWithOptuna(XGBoostModel):
    """
    XGBoost model with Optuna hyperparameter optimization.
    """
    
    def __init__(
        self,
        name: str = 'xgboost_optuna',
        params: Optional[Dict[str, Any]] = None,
        n_trials: int = 50,
        timeout: int = 3600
    ):
        """
        Initialize XGBoost model with Optuna support.
        
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
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'tree_method': 'hist',
                'n_jobs': -1,
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            
            return rmse
        
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials...")
        
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
        self.model = xgb.XGBRegressor(**self.params)
        
        return self.best_params

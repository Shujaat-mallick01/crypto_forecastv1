"""
Ensemble model combining multiple prediction models.

Provides ensemble methods:
- Weighted averaging
- Stacking
- Voting
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .base import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining XGBoost and LightGBM predictions.
    
    Ensemble methods:
    - weighted_average: Simple weighted average of predictions
    - stacking: Train a meta-model on base model predictions
    """
    
    def __init__(
        self,
        name: str = 'ensemble',
        method: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None,
        xgboost_params: Optional[Dict[str, Any]] = None,
        lightgbm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            name: Model name
            method: Ensemble method ('weighted_average' or 'stacking')
            weights: Model weights for weighted average
            xgboost_params: XGBoost hyperparameters
            lightgbm_params: LightGBM hyperparameters
        """
        super().__init__(name, {
            'method': method,
            'weights': weights or {'xgboost': 0.5, 'lightgbm': 0.5}
        })
        
        self.method = method
        self.weights = weights or {'xgboost': 0.5, 'lightgbm': 0.5}
        
        # Initialize base models
        self.models = {
            'xgboost': XGBoostModel('xgboost', xgboost_params),
            'lightgbm': LightGBMModel('lightgbm', lightgbm_params)
        }
        
        # Meta-model for stacking
        self.meta_model = None
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'EnsembleModel':
        """
        Train all base models and optionally the meta-model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble model ({self.method})...")
        
        self.feature_names = list(X_train.columns)
        
        # Train each base model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train, X_val, y_val)
        
        # For stacking, train meta-model
        if self.method == 'stacking' and X_val is not None:
            self._train_meta_model(X_train, y_train, X_val, y_val)
        
        # Compute combined feature importance
        self._compute_ensemble_importance()
        
        self.is_fitted = True
        self.metadata['n_samples'] = len(X_train)
        self.metadata['n_features'] = len(self.feature_names)
        self.metadata['base_models'] = list(self.models.keys())
        
        logger.info("Ensemble training complete")
        
        return self
    
    def _train_meta_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """
        Train meta-model for stacking.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        from sklearn.linear_model import Ridge
        
        logger.info("Training stacking meta-model...")
        
        # Get base model predictions on validation set
        meta_features = self._get_meta_features(X_val)
        
        # Train Ridge regression as meta-model
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y_val)
        
        # Update weights based on meta-model coefficients
        coefs = self.meta_model.coef_
        total = sum(abs(c) for c in coefs)
        
        for i, name in enumerate(self.models.keys()):
            self.weights[name] = abs(coefs[i]) / total
        
        logger.info(f"Stacking weights: {self.weights}")
    
    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from base models as meta-features.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, n_models)
        """
        predictions = []
        
        for name, model in self.models.items():
            preds = model.predict(X)
            predictions.append(preds)
        
        return np.column_stack(predictions)
    
    def _compute_ensemble_importance(self) -> None:
        """Compute weighted average of feature importances."""
        importances = np.zeros(len(self.feature_names))
        
        for name, model in self.models.items():
            if model.feature_importances_ is not None:
                weight = self.weights.get(name, 1.0 / len(self.models))
                importances += weight * model.feature_importances_
        
        self.feature_importances_ = importances
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.method == 'weighted_average':
            return self._predict_weighted_average(X)
        elif self.method == 'stacking':
            return self._predict_stacking(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def _predict_weighted_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate weighted average predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0 / len(self.models))
            predictions += weight * model.predict(X)
        
        return predictions
    
    def _predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate stacking predictions using meta-model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if self.meta_model is None:
            logger.warning("Meta-model not trained, using weighted average")
            return self._predict_weighted_average(X)
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def get_individual_predictions(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base model.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping model names to predictions
        """
        return {
            name: model.predict(X)
            for name, model in self.models.items()
        }
    
    def get_model_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for each base model.
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Dictionary of metrics for each model
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        metrics = {}
        
        for name, model in self.models.items():
            preds = model.predict(X)
            
            metrics[name] = {
                'mae': mean_absolute_error(y, preds),
                'rmse': np.sqrt(mean_squared_error(y, preds)),
                'mape': np.mean(np.abs((y - preds) / y)) * 100
            }
        
        # Ensemble metrics
        ensemble_preds = self.predict(X)
        metrics['ensemble'] = {
            'mae': mean_absolute_error(y, ensemble_preds),
            'rmse': np.sqrt(mean_squared_error(y, ensemble_preds)),
            'mape': np.mean(np.abs((y - ensemble_preds) / y)) * 100
        }
        
        return metrics
    
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        Load ensemble model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded EnsembleModel instance
        """
        from pathlib import Path
        import joblib
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        # Extract params
        params = save_data.get('params', {})
        
        # Create instance with correct arguments for EnsembleModel
        instance = cls(
            name=save_data.get('metadata', {}).get('name', 'ensemble'),
            method=params.get('method', 'weighted_average'),
            weights=params.get('weights', {'xgboost': 0.5, 'lightgbm': 0.5})
        )
        
        # Restore the trained model and attributes
        # instance.models = save_data.get('model', instance.models)
        saved_models = save_data.get('model')
        if saved_models is not None and isinstance(saved_models, dict):
            instance.models = saved_models
        if instance.models is None:
            logger.warning("base models not found in saved file")
            instance.is_fitted = False
        instance.feature_names = save_data.get('feature_names', [])
        instance.feature_importances_ = save_data.get('feature_importances')
        instance.metadata = save_data.get('metadata', {})
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance
    def save(self, filepath: str) -> str:
        """
        Save ensemble model to file.
        
        Args:
            filepath: Path to save model
            
        Returns:
            Path to saved file
        """
        from pathlib import Path
        import joblib
        from datetime import datetime
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata['saved_at'] = datetime.now().isoformat()
        self.metadata['feature_count'] = len(self.feature_names)
        
        # Save models dict and metadata
        save_data = {
            'model': self.models,  # Save the base models dict
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'metadata': self.metadata,
            'meta_model': self.meta_model
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
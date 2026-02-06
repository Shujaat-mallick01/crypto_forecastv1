"""
Base model interface for all prediction models.

Defines the common interface that all models must implement.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for prediction models.
    
    All model implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            name: Model name identifier
            params: Model hyperparameters
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'name': name
        }
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        pass
    
    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Train and predict in one step.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importances_ is None:
            logger.warning("Feature importances not available")
            return pd.DataFrame()
        
        names = feature_names or self.feature_names
        
        if len(names) != len(self.feature_importances_):
            logger.warning("Feature names length mismatch")
            names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': names,
            'importance': self.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, filepath: str) -> str:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            
        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata['saved_at'] = datetime.now().isoformat()
        self.metadata['feature_count'] = len(self.feature_names)
        
        # Save model and metadata
        save_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'metadata': self.metadata
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            name=save_data['metadata'].get('name', 'loaded_model'),
            params=save_data['params']
        )
        
        instance.model = save_data['model']
        instance.feature_names = save_data['feature_names']
        instance.feature_importances_ = save_data['feature_importances']
        instance.metadata = save_data['metadata']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"

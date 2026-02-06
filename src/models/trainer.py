"""
Model training orchestration module.

Handles:
- Time-series cross-validation
- Model training for all targets and horizons
- Model persistence
- Training metrics collection
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import joblib

from .base import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .ensemble import EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training orchestrator.
    
    Manages training workflow including:
    - Temporal train/test splitting
    - Time-series cross-validation
    - Multi-target, multi-horizon training
    - Model persistence
    """
    
    def __init__(self, config):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_config = config.model_config
        
        # Ensure models directory exists
        config.models_path.mkdir(parents=True, exist_ok=True)
    
    def temporal_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data temporally (no shuffling).
        
        Args:
            X: Features
            y: Targets
            test_size: Test set fraction
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self.model_config.test_size
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Temporal split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def cross_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of metric lists across folds
        """
        from ..evaluation.metrics import calculate_metrics
        
        n_splits = n_splits or self.model_config.validation_splits
        
        logger.info(f"Time-series cross-validation with {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {'mae': [], 'rmse': [], 'mape': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]
            
            # Clone model for this fold
            if isinstance(model, XGBoostModel):
                fold_model = XGBoostModel(f"{model.name}_fold{fold}", model.params)
            elif isinstance(model, LightGBMModel):
                fold_model = LightGBMModel(f"{model.name}_fold{fold}", model.params)
            elif isinstance(model, EnsembleModel):
                fold_model = EnsembleModel(
                    f"{model.name}_fold{fold}",
                    method=model.method,
                    weights=model.weights
                )
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            # Train and evaluate
            fold_model.fit(X_train_cv, y_train_cv)
            y_pred = fold_model.predict(X_val_cv)
            
            metrics = calculate_metrics(y_val_cv, y_pred)
            
            for metric, value in metrics.items():
                if metric in cv_scores:
                    cv_scores[metric].append(value)
            
            logger.debug(f"Fold {fold}: {metrics}")
        
        # Log summary
        for metric, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"CV {metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")
        
        return cv_scores
    
    def create_model(
        self,
        algorithm: Optional[str] = None,
        use_ensemble: bool = False
    ) -> BaseModel:
        """
        Create a model instance based on configuration.
        
        Args:
            algorithm: Algorithm name ('xgboost', 'lightgbm')
            use_ensemble: Whether to use ensemble model
            
        Returns:
            Model instance
        """
        algorithm = algorithm or self.model_config.default_algorithm
        
        if use_ensemble or self.model_config.ensemble_enabled:
            return EnsembleModel(
                name='ensemble',
                method='weighted_average',
                weights=self.model_config.ensemble_weights,
                xgboost_params=self.model_config.xgboost_params,
                lightgbm_params=self.model_config.lightgbm_params
            )
        
        if algorithm == 'xgboost':
            return XGBoostModel('xgboost', self.model_config.xgboost_params)
        elif algorithm == 'lightgbm':
            return LightGBMModel('lightgbm', self.model_config.lightgbm_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        algorithm: Optional[str] = None,
        use_ensemble: bool = False
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train a single model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            algorithm: Algorithm to use
            use_ensemble: Whether to use ensemble
            
        Returns:
            Tuple of (trained model, test metrics)
        """
        from ..evaluation.metrics import calculate_metrics
        
        # Create model
        model = self.create_model(algorithm, use_ensemble)
        
        # Train
        model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred)
            logger.info(f"Validation metrics: {metrics}")
        
        return model, metrics
    
    def train_for_target_horizon(
        self,
        df: pd.DataFrame,
        symbol: str,
        target: str,
        horizon: int,
        feature_pipeline
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train model for specific target and horizon.
        
        Args:
            df: Featured DataFrame
            symbol: Coin symbol
            target: Target type ('price' or 'market_cap')
            horizon: Forecast horizon in days
            feature_pipeline: Feature pipeline instance
            
        Returns:
            Tuple of (trained model, metrics)
        """
        logger.info(f"Training {target}@{horizon}d for {symbol}...")
        
        # Prepare data
        X, y, feature_names = feature_pipeline.prepare_training_data(
            df, target, horizon
        )
        
        # Temporal split
        X_train, X_test, y_train, y_test = self.temporal_split(X, y)
        
        # Train model
        model, metrics = self.train_model(
            X_train, y_train, X_test, y_test
        )
        
        # Save model
        model_name = f"{self.config.model_prefix}_{symbol}_{target}_{horizon}d"
        model_path = self.save_model(model, model_name)
        
        return model, metrics
    
    def train_all_models(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_pipeline
    ) -> Dict[str, Dict]:
        """
        Train all models for all targets and horizons.
        
        Args:
            df: Featured DataFrame
            symbol: Coin symbol
            feature_pipeline: Feature pipeline instance
            
        Returns:
            Dictionary of results for each target/horizon
        """
        results = {}
        
        for target in self.model_config.targets:
            for horizon in self.model_config.horizons:
                key = f"{target}_{horizon}d"
                
                try:
                    model, metrics = self.train_for_target_horizon(
                        df, symbol, target, horizon, feature_pipeline
                    )
                    
                    results[key] = {
                        'model': model,
                        'metrics': metrics,
                        'target': target,
                        'horizon': horizon
                    }
                    
                    logger.info(f"✓ {key}: RMSE={metrics.get('rmse', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"✗ {key}: {e}")
                    results[key] = {'error': str(e)}
        
        return results
    
    def save_model(self, model: BaseModel, model_name: str) -> Path:
        """
        Save model to disk.
        
        Args:
            model: Trained model
            model_name: Model name/identifier
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = self.config.models_path / filename
        
        model.save(str(filepath))
        
        # Cleanup old models
        self._cleanup_old_models(model_name)
        
        return filepath
    
    def load_model(self, model_name: str) -> BaseModel:
        """
        Load latest model with given name.
        
        Args:
            model_name: Model name/identifier
            
        Returns:
            Loaded model instance
        """
        pattern = f"{model_name}_*.joblib"
        files = list(self.config.models_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No models found for {model_name}")
        
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        
        # Determine model type from file
        data = joblib.load(latest_file)
        model_type = data.get('metadata', {}).get('name', 'xgboost')
        
        if 'ensemble' in model_type:
            return EnsembleModel.load(str(latest_file))
        elif 'lightgbm' in model_type:
            return LightGBMModel.load(str(latest_file))
        else:
            return XGBoostModel.load(str(latest_file))
    
    def _cleanup_old_models(self, model_name: str) -> None:
        """Remove old model versions."""
        keep_n = self.config.keep_last_n_models
        
        pattern = f"{model_name}_*.joblib"
        files = sorted(
            self.config.models_path.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_file in files[keep_n:]:
            old_file.unlink()
            logger.debug(f"Removed old model: {old_file}")

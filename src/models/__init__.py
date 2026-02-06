"""
Models module for cryptocurrency price prediction.

This module provides:
- Base model interface
- XGBoost model implementation
- LightGBM model implementation
- Ensemble model combining multiple algorithms
- Hyperparameter optimization with Optuna
"""

from .base import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .ensemble import EnsembleModel
from .trainer import ModelTrainer

__all__ = [
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "EnsembleModel",
    "ModelTrainer"
]

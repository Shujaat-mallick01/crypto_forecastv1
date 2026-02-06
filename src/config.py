"""
Configuration management module.

Provides centralized configuration loading and validation.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class CoinConfig:
    """Configuration for a single cryptocurrency."""
    symbol: str
    coingecko_id: str


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lag_periods: List[int]
    rolling_windows: List[int]
    return_periods: List[int]
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    scaling_method: str = "robust"
    clip_outliers: bool = True
    outlier_threshold: float = 5.0


@dataclass
class ModelConfig:
    """Model training configuration."""
    targets: List[str]
    horizons: List[int]
    test_size: float
    validation_splits: int
    random_state: int
    default_algorithm: str
    xgboost_params: Dict[str, Any]
    lightgbm_params: Dict[str, Any]
    ensemble_enabled: bool = True
    ensemble_weights: Dict[str, float] = field(default_factory=dict)


class Config:
    """
    Central configuration manager.
    
    Loads configuration from YAML file and provides typed access
    to all settings with validation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._load_config()
        self._setup_logging()
        self._validate_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_config = self._raw_config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/pipeline.log')
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        required_sections = ['api', 'data', 'features', 'model', 'storage']
        
        for section in required_sections:
            if section not in self._raw_config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate data paths
        data_config = self._raw_config['data']
        for path_key in ['raw_path', 'processed_path', 'predictions_path']:
            if path_key not in data_config:
                raise ValueError(f"Missing data path: {path_key}")
        
        logger.info("Configuration validation passed")
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    
    @property
    def coingecko_base_url(self) -> str:
        return self._raw_config['api']['coingecko']['base_url']
    
    @property
    def api_timeout(self) -> int:
        return self._raw_config['api']['coingecko']['timeout']
    
    @property
    def api_max_retries(self) -> int:
        return self._raw_config['api']['coingecko']['max_retries']
    
    @property
    def api_backoff_factor(self) -> float:
        return self._raw_config['api']['coingecko']['backoff_factor']
    
    @property
    def rate_limit_delay(self) -> float:
        return self._raw_config['api']['coingecko'].get('rate_limit_delay', 1.5)
    
    # ==========================================================================
    # Data Configuration
    # ==========================================================================
    
    @property
    def raw_data_path(self) -> Path:
        return Path(self._raw_config['data']['raw_path'])
    
    @property
    def processed_data_path(self) -> Path:
        return Path(self._raw_config['data']['processed_path'])
    
    @property
    def predictions_path(self) -> Path:
        return Path(self._raw_config['data']['predictions_path'])
    
    @property
    def reports_path(self) -> Path:
        return Path(self._raw_config['data']['reports_path'])
    
    @property
    def visualizations_path(self) -> Path:
        return Path(self._raw_config['data']['visualizations_path'])
    
    @property
    def coins(self) -> List[CoinConfig]:
        """Get list of configured cryptocurrencies."""
        return [
            CoinConfig(symbol=c['symbol'], coingecko_id=c['coingecko_id'])
            for c in self._raw_config['data']['coins']
        ]
    
    @property
    def coin_symbols(self) -> List[str]:
        """Get list of coin symbols."""
        return [c.symbol for c in self.coins]
    
    def get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko ID for a symbol."""
        for coin in self.coins:
            if coin.symbol == symbol:
                return coin.coingecko_id
        return None
    
    @property
    def historical_days(self) -> int:
        return self._raw_config['data']['historical_days']
    
    @property
    def min_data_points(self) -> int:
        return self._raw_config['data'].get('min_data_points', 100)
    
    # ==========================================================================
    # Feature Configuration
    # ==========================================================================
    
    @property
    def feature_config(self) -> FeatureConfig:
        """Get feature engineering configuration."""
        feat = self._raw_config['features']
        tech = feat.get('technical', {})
        scaling = feat.get('scaling', {})
        
        return FeatureConfig(
            lag_periods=feat['lag_periods'],
            rolling_windows=feat['rolling_windows'],
            return_periods=feat['return_periods'],
            rsi_period=tech.get('rsi_period', 14),
            macd_fast=tech.get('macd_fast', 12),
            macd_slow=tech.get('macd_slow', 26),
            macd_signal=tech.get('macd_signal', 9),
            bollinger_period=tech.get('bollinger_period', 20),
            bollinger_std=tech.get('bollinger_std', 2.0),
            scaling_method=scaling.get('method', 'robust'),
            clip_outliers=scaling.get('clip_outliers', True),
            outlier_threshold=scaling.get('outlier_threshold', 5.0)
        )
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    
    @property
    def model_config(self) -> ModelConfig:
        """Get model training configuration."""
        model = self._raw_config['model']
        ensemble = model.get('ensemble', {})
        
        return ModelConfig(
            targets=model['targets'],
            horizons=model['horizons'],
            test_size=model['test_size'],
            validation_splits=model['validation_splits'],
            random_state=model['random_state'],
            default_algorithm=model['default_algorithm'],
            xgboost_params=model['xgboost'],
            lightgbm_params=model['lightgbm'],
            ensemble_enabled=ensemble.get('enabled', True),
            ensemble_weights=ensemble.get('weights', {'xgboost': 0.5, 'lightgbm': 0.5})
        )
    
    @property
    def targets(self) -> List[str]:
        return self._raw_config['model']['targets']
    
    @property
    def horizons(self) -> List[int]:
        return self._raw_config['model']['horizons']
    
    # ==========================================================================
    # Storage Configuration
    # ==========================================================================
    
    @property
    def models_path(self) -> Path:
        return Path(self._raw_config['storage']['models_path'])
    
    @property
    def model_prefix(self) -> str:
        return self._raw_config['storage']['model_prefix']
    
    @property
    def keep_last_n_models(self) -> int:
        return self._raw_config['storage'].get('keep_last_n', 3)
    
    # ==========================================================================
    # Parallel Processing
    # ==========================================================================
    
    @property
    def parallel_enabled(self) -> bool:
        return self._raw_config.get('parallel', {}).get('enabled', True)
    
    @property
    def max_workers(self) -> int:
        workers = self._raw_config.get('parallel', {}).get('max_workers', -1)
        if workers == -1:
            import multiprocessing
            return multiprocessing.cpu_count()
        return workers
    
    # ==========================================================================
    # Hyperparameter Tuning
    # ==========================================================================
    
    @property
    def hyperopt_enabled(self) -> bool:
        return self._raw_config.get('hyperopt', {}).get('enabled', False)
    
    @property
    def hyperopt_trials(self) -> int:
        return self._raw_config.get('hyperopt', {}).get('n_trials', 50)
    
    @property
    def hyperopt_timeout(self) -> int:
        return self._raw_config.get('hyperopt', {}).get('timeout', 3600)
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================
    
    def create_directories(self) -> None:
        """Create all required directories."""
        directories = [
            self.raw_data_path,
            self.processed_data_path,
            self.predictions_path,
            self.reports_path,
            self.visualizations_path,
            self.models_path,
            Path('logs')
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("All directories created")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return raw configuration dictionary."""
        return self._raw_config.copy()
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, coins={len(self.coins)})"


# Convenience function for loading config
def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration from file."""
    return Config(config_path)

import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import Config
from src.pipeline.orchestrator import Pipeline

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'
        self.config = Config(str(config_path))
        self.models_dir = self.config.models_path

    def train_model(self, coin_symbols, model_type='ensemble', skip_ingestion=False):
        """
        Train models for specified coins using the actual ML pipeline.

        Args:
            coin_symbols: List of coin symbols or single symbol string
            model_type: 'ensemble', 'xgboost', or 'lightgbm'
            skip_ingestion: If True, skip data fetching and use existing data

        Returns:
            Dict with success status and metrics
        """
        try:
            if isinstance(coin_symbols, str):
                coin_symbols = [coin_symbols.upper()]
            else:
                coin_symbols = [s.upper() for s in coin_symbols]

            logger.info(f"Starting training for {coin_symbols} with {model_type}")

            # Create a fresh pipeline instance for this training run
            pipeline = Pipeline(self.config)

            # Run the full pipeline
            result = pipeline.run(
                symbols=coin_symbols,
                skip_ingestion=skip_ingestion,
                skip_training=False,
                parallel=True
            )

            if result.get('success'):
                # Extract metrics from training results
                metrics = self._extract_metrics(result)
                return {
                    'success': True,
                    'metrics': metrics,
                    'duration': result.get('duration_seconds', 0),
                    'symbols': coin_symbols,
                    'steps': {
                        step: {'success': data.get('success', False)}
                        for step, data in result.get('steps', {}).items()
                    }
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Pipeline failed'),
                    'steps': {
                        step: {
                            'success': data.get('success', False),
                            'error': data.get('error')
                        }
                        for step, data in result.get('steps', {}).items()
                    }
                }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def train_only(self, coin_symbols, model_type='ensemble'):
        """Train using existing data (skip ingestion)."""
        return self.train_model(coin_symbols, model_type, skip_ingestion=True)

    def ingest_data(self, coin_symbols):
        """Only fetch/refresh data for coins."""
        try:
            if isinstance(coin_symbols, str):
                coin_symbols = [coin_symbols.upper()]

            pipeline = Pipeline(self.config)
            result = pipeline.ingest(coin_symbols)

            return {
                'success': result.get('success', False),
                'coins': result.get('coins', {}),
                'error': result.get('error')
            }
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _extract_metrics(self, pipeline_result):
        """Extract training metrics from pipeline result."""
        metrics = {}
        training_data = pipeline_result.get('steps', {}).get('training', {})
        models = training_data.get('models', {})

        for symbol, symbol_data in models.items():
            if isinstance(symbol_data, dict) and 'error' not in symbol_data:
                symbol_metrics = {}
                for key, val in symbol_data.items():
                    if isinstance(val, dict) and 'metrics' in val:
                        m = val['metrics']
                        symbol_metrics[key] = {
                            k: round(v, 6) if isinstance(v, float) else v
                            for k, v in m.items()
                        }
                if symbol_metrics:
                    metrics[symbol] = symbol_metrics

        return metrics

    def get_model_metrics(self, coin_symbol):
        """Get metrics for trained models of a coin."""
        coin_symbol = coin_symbol.upper()
        metrics = {}

        for target in self.config.targets:
            for horizon in self.config.horizons:
                model_name = f"{self.config.model_prefix}_{coin_symbol}_{target}_{horizon}d"
                pattern = f"{model_name}_*.joblib"
                files = list(self.models_dir.glob(pattern))

                if files:
                    latest = max(files, key=lambda p: p.stat().st_mtime)
                    try:
                        import joblib
                        data = joblib.load(latest)
                        model_meta = data.get('metadata', {})
                        metrics[f"{target}_{horizon}d"] = {
                            'model_name': model_meta.get('name', 'unknown'),
                            'trained_at': model_meta.get('saved_at', ''),
                            'n_features': model_meta.get('n_features', 0),
                            'n_samples': model_meta.get('n_samples', 0),
                            'file': latest.name
                        }
                    except Exception as e:
                        logger.error(f"Error loading model metadata: {e}")
                        metrics[f"{target}_{horizon}d"] = {'file': latest.name}

        return metrics

    def get_training_status_summary(self):
        """Get a summary of available models and data."""
        summary = {
            'coins_configured': self.config.coin_symbols,
            'targets': self.config.targets,
            'horizons': self.config.horizons,
            'models_available': {},
            'data_available': {}
        }

        from src.data.loader import DataLoader
        loader = DataLoader(self.config)

        for coin in self.config.coin_symbols:
            # Check data
            raw_df = loader.load_raw_data(coin)
            proc_df = loader.load_processed_data(coin)
            summary['data_available'][coin] = {
                'raw': raw_df is not None,
                'processed': proc_df is not None,
                'raw_rows': len(raw_df) if raw_df is not None else 0,
                'processed_rows': len(proc_df) if proc_df is not None else 0
            }

            # Check models
            coin_models = {}
            for target in self.config.targets:
                for horizon in self.config.horizons:
                    model_name = f"{self.config.model_prefix}_{coin}_{target}_{horizon}d"
                    pattern = f"{model_name}_*.joblib"
                    files = list(self.models_dir.glob(pattern))
                    coin_models[f"{target}_{horizon}d"] = len(files) > 0
            summary['models_available'][coin] = coin_models

        return summary

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import Config
from src.data.loader import DataLoader
from src.features.pipeline import FeaturePipeline
from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self):
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'
        self.config = Config(str(config_path))
        self.loader = DataLoader(self.config)
        self.feature_pipeline = FeaturePipeline(self.config)
        self.trainer = ModelTrainer(self.config)
        self.models_dir = self.config.models_path

    def get_available_models(self):
        """Get list of trained models available for prediction."""
        available = {}
        for coin in self.config.coin_symbols:
            coin_models = {}
            for target in self.config.targets:
                for horizon in self.config.horizons:
                    model_name = f"{self.config.model_prefix}_{coin}_{target}_{horizon}d"
                    pattern = f"{model_name}_*.joblib"
                    files = list(self.models_dir.glob(pattern))
                    if files:
                        latest = max(files, key=lambda p: p.stat().st_mtime)
                        coin_models[f"{target}_{horizon}d"] = {
                            'path': str(latest),
                            'modified': datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
                        }
            if coin_models:
                available[coin] = coin_models
        return available

    def predict(self, coin_symbol, prediction_type='price', horizon=None):
        """Generate prediction for a coin using trained models."""
        try:
            coin_symbol = coin_symbol.upper()
            horizons = [horizon] if horizon else self.config.horizons

            # Load processed data
            df = self.loader.load_processed_data(coin_symbol)
            if df is None:
                return {
                    'success': False,
                    'error': f'No processed data for {coin_symbol}. Run training first.'
                }

            # Get feature columns and latest data
            feature_cols = self.feature_pipeline.get_feature_names(df)
            X_latest = df[feature_cols].iloc[[-1]]

            # Get current values
            current_price = float(df.iloc[-1].get('price', df.iloc[-1].get('close', 0)))
            current_market_cap = float(df.iloc[-1].get('market_cap', 0))

            predictions = []
            for h in horizons:
                if prediction_type in ['price', 'all']:
                    pred = self._predict_single(
                        coin_symbol, 'price', h, X_latest, current_price
                    )
                    if pred:
                        predictions.append(pred)

                if prediction_type in ['market_cap', 'all']:
                    pred = self._predict_single(
                        coin_symbol, 'market_cap', h, X_latest, current_market_cap
                    )
                    if pred:
                        predictions.append(pred)

            if not predictions:
                return {
                    'success': False,
                    'error': f'No trained models found for {coin_symbol}. Run training first.'
                }

            return {
                'success': True,
                'symbol': coin_symbol,
                'current_price': current_price,
                'current_market_cap': current_market_cap,
                'predictions': predictions,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error for {coin_symbol}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _predict_single(self, symbol, target, horizon, X_latest, current_value):
        """Predict a single target/horizon combination."""
        try:
            model_name = f"{self.config.model_prefix}_{symbol}_{target}_{horizon}d"
            model = self.trainer.load_model(model_name)

            pred_value = float(model.predict(X_latest)[0])
            change_pct = (pred_value - current_value) / current_value * 100 if current_value else 0

            return {
                'target': target,
                'horizon': horizon,
                'current_value': current_value,
                'predicted_value': pred_value,
                'change_pct': round(change_pct, 2),
                'future_date': (datetime.now() + timedelta(days=horizon)).isoformat(),
                'model_version': model.name if hasattr(model, 'name') else 'ensemble'
            }
        except FileNotFoundError:
            logger.warning(f"No model found for {symbol} {target}@{horizon}d")
            return None
        except Exception as e:
            logger.error(f"Prediction failed for {symbol} {target}@{horizon}d: {e}")
            return None

    def get_available_coins(self):
        """Get list of available coins."""
        return self.config.coin_symbols

    def get_coins_with_models(self):
        """Get coins that have trained models."""
        available = self.get_available_models()
        return list(available.keys())

    def get_coins_with_data(self):
        """Get coins that have processed data available."""
        coins_with_data = []
        for coin in self.config.coin_symbols:
            if self.loader.load_processed_data(coin) is not None:
                coins_with_data.append(coin)
        return coins_with_data

    def get_live_market_data(self):
        """Get latest data for all coins from processed data files."""
        market_data = {}
        for coin in self.config.coin_symbols:
            df = self.loader.load_processed_data(coin)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                price = float(latest.get('price', latest.get('close', 0)))
                prev_price = float(df.iloc[-2].get('price', df.iloc[-2].get('close', 0))) if len(df) > 1 else price
                change_24h = ((price - prev_price) / prev_price * 100) if prev_price else 0

                market_data[coin] = {
                    'price': price,
                    'market_cap': float(latest.get('market_cap', 0)),
                    'volume': float(latest.get('volume', latest.get('total_volume', 0))),
                    'change_24h': round(change_24h, 2),
                    'timestamp': str(latest.get('timestamp', datetime.now().isoformat()))
                }
            else:
                market_data[coin] = {
                    'price': 0,
                    'market_cap': 0,
                    'volume': 0,
                    'change_24h': 0,
                    'timestamp': datetime.now().isoformat(),
                    'no_data': True
                }
        return market_data

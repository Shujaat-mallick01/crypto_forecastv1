"""
Prediction pipeline for generating forecasts.

Handles:
- Loading trained models
- Preparing latest data
- Generating multi-horizon predictions
- Saving and visualizing results
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from ..config import Config
from ..data import DataLoader
from ..features import FeaturePipeline
from ..models import ModelTrainer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Prediction generation pipeline.
    
    Generates forecasts using trained models for all
    configured targets and horizons.
    """
    
    def __init__(self, config: Config):
        """
        Initialize prediction pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.loader = DataLoader(config)
        self.feature_pipeline = FeaturePipeline(config)
        self.trainer = ModelTrainer(config)
    
    def predict_all_horizons(self, symbol: str) -> Dict[str, Any]:
        """
        Generate predictions for all targets and horizons.
        
        Args:
            symbol: Coin symbol
            
        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Generating predictions for {symbol}...")
        
        # Load processed data
        df = self.loader.load_processed_data(symbol)
        
        if df is None:
            raise ValueError(f"No processed data for {symbol}")
        
        # Get latest data point
        latest = df.iloc[-1]
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_timestamp': latest['timestamp'] if 'timestamp' in df.columns else None,
            'current_price': latest.get('price', latest.get('close')),
            'current_market_cap': latest.get('market_cap'),
            'predictions': []
        }
        
        # Get feature columns
        feature_cols = self.feature_pipeline.get_feature_names(df)
        X_latest = df[feature_cols].iloc[[-1]]
        
        # Generate predictions for each target and horizon
        for target in self.config.targets:
            current_value = results[f'current_{target}']
            
            for horizon in self.config.horizons:
                try:
                    # Load model
                    model_name = f"{self.config.model_prefix}_{symbol}_{target}_{horizon}d"
                    model = self.trainer.load_model(model_name)
                    
                    # Predict
                    pred_value = model.predict(X_latest)[0]
                    
                    # Calculate change
                    change_pct = (pred_value - current_value) / current_value * 100
                    
                    results['predictions'].append({
                        'target': target,
                        'horizon': horizon,
                        'current_value': current_value,
                        'predicted_value': pred_value,
                        'change_pct': change_pct,
                        'future_date': (
                            pd.to_datetime(latest['timestamp']) + timedelta(days=horizon)
                            if 'timestamp' in df.columns else None
                        )
                    })
                    
                    logger.debug(f"{target}@{horizon}d: ${pred_value:,.2f} ({change_pct:+.2f}%)")
                    
                except FileNotFoundError:
                    logger.warning(f"Model not found for {target}@{horizon}d")
                except Exception as e:
                    logger.error(f"Prediction failed for {target}@{horizon}d: {e}")
        
        return results
    
    def predict_multiple(
        self,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple coins.
        
        Args:
            symbols: List of symbols (default: all configured)
            
        Returns:
            DataFrame with all predictions
        """
        symbols = symbols or self.config.coin_symbols
        all_predictions = []
        
        for symbol in symbols:
            try:
                results = self.predict_all_horizons(symbol)
                
                for pred in results['predictions']:
                    all_predictions.append({
                        'symbol': symbol,
                        'generated_at': results['timestamp'],
                        **pred
                    })
                    
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
        
        return pd.DataFrame(all_predictions)
    
    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save predictions to file.
        
        Args:
            predictions_df: DataFrame with predictions
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        self.config.predictions_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"predictions_{timestamp}.csv"
        
        filepath = self.config.predictions_path / filename
        predictions_df.to_csv(filepath, index=False)
        
        logger.info(f"Predictions saved to {filepath}")
        
        return filepath
    
    def get_forecast_summary(
        self,
        symbol: str
    ) -> pd.DataFrame:
        """
        Get formatted forecast summary for a coin.
        
        Args:
            symbol: Coin symbol
            
        Returns:
            DataFrame with forecast summary
        """
        results = self.predict_all_horizons(symbol)
        
        summary_data = []
        
        for pred in results['predictions']:
            summary_data.append({
                'Target': pred['target'].title(),
                'Horizon': f"{pred['horizon']} days",
                'Current': f"${pred['current_value']:,.2f}",
                'Predicted': f"${pred['predicted_value']:,.2f}",
                'Change': f"{pred['change_pct']:+.2f}%",
                'Direction': '📈' if pred['change_pct'] > 0 else '📉'
            })
        
        return pd.DataFrame(summary_data)
    
    def generate_report(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive forecast report.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Report dictionary
        """
        symbols = symbols or self.config.coin_symbols
        
        report = {
            'generated_at': datetime.now(),
            'coins': {}
        }
        
        predictions_df = self.predict_multiple(symbols)
        
        for symbol in symbols:
            symbol_preds = predictions_df[predictions_df['symbol'] == symbol]
            
            if symbol_preds.empty:
                continue
            
            report['coins'][symbol] = {
                'price_1d': symbol_preds[
                    (symbol_preds['target'] == 'price') & 
                    (symbol_preds['horizon'] == 1)
                ]['predicted_value'].iloc[0] if len(symbol_preds) > 0 else None,
                
                'price_7d': symbol_preds[
                    (symbol_preds['target'] == 'price') & 
                    (symbol_preds['horizon'] == 7)
                ]['predicted_value'].iloc[0] if len(symbol_preds) > 0 else None,
                
                'price_30d': symbol_preds[
                    (symbol_preds['target'] == 'price') & 
                    (symbol_preds['horizon'] == 30)
                ]['predicted_value'].iloc[0] if len(symbol_preds) > 0 else None,
                
                'predictions': symbol_preds.to_dict('records')
            }
        
        # Save report
        self.config.reports_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.config.reports_path / f"forecast_report_{timestamp}.csv"
        predictions_df.to_csv(report_path, index=False)
        
        report['report_path'] = str(report_path)
        
        return report

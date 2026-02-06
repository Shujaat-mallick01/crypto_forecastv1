"""
Pipeline orchestrator for end-to-end ML workflow.

Manages the complete workflow:
1. Data ingestion
2. Data validation
3. Feature engineering
4. Model training
5. Prediction generation
6. Evaluation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..config import Config
from ..data import DataIngestion, DataLoader, DataValidator
from ..features import FeaturePipeline
from ..models import ModelTrainer
from ..evaluation import MetricsCalculator, PredictionVisualizer

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline orchestrator.
    
    Coordinates all stages of the ML pipeline from data ingestion
    to prediction generation.
    """
    
    def __init__(self, config: Optional[Config] = None, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object (or load from path)
            config_path: Path to config file
        """
        self.config = config or Config(config_path)
        self.config.create_directories()
        
        # Initialize components
        self.ingestion = DataIngestion(self.config)
        self.loader = DataLoader(self.config)
        self.validator = DataValidator(self.config)
        self.feature_pipeline = FeaturePipeline(self.config)
        self.trainer = ModelTrainer(self.config)
        self.metrics = MetricsCalculator()
        self.visualizer = PredictionVisualizer(self.config.visualizations_path)
        
        # Track execution
        self.execution_log: List[Dict] = []
    
    def run(
        self,
        symbols: Optional[List[str]] = None,
        skip_ingestion: bool = False,
        skip_training: bool = False,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            symbols: List of coin symbols (default: all configured)
            skip_ingestion: Skip data fetching
            skip_training: Skip model training (use existing models)
            parallel: Enable parallel processing
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        symbols = symbols or self.config.coin_symbols
        
        logger.info("=" * 80)
        logger.info("CRYPTO FORECAST PIPELINE")
        logger.info(f"Start time: {start_time}")
        logger.info(f"Coins: {symbols}")
        logger.info("=" * 80)
        
        results = {
            'start_time': start_time,
            'symbols': symbols,
            'steps': {}
        }
        
        try:
            # Step 1: Data Ingestion
            if not skip_ingestion:
                results['steps']['ingestion'] = self._run_ingestion(symbols)
            else:
                logger.info("Skipping data ingestion")
            
            # Step 2: Data Validation
            results['steps']['validation'] = self._run_validation(symbols)
            
            # Step 3: Feature Engineering
            results['steps']['features'] = self._run_feature_engineering(symbols, parallel)
            
            # Step 4: Model Training
            if not skip_training:
                results['steps']['training'] = self._run_training(symbols, parallel)
            else:
                logger.info("Skipping model training")
            
            # Step 5: Generate Predictions
            results['steps']['predictions'] = self._run_predictions(symbols)
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['success'] = False
            results['error'] = str(e)
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results['end_time'] = end_time
        results['duration_seconds'] = duration
        
        self._print_summary(results)
        
        return results
    
    def _run_ingestion(self, symbols: List[str]) -> Dict:
        """Run data ingestion step."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 60)
        
        result = {'success': False, 'coins': {}}
        
        try:
            data = self.ingestion.fetch_all_coins(symbols)
            
            result['success'] = True
            result['coins'] = {
                symbol: len(df) for symbol, df in data.items()
            }
            
            logger.info(f"✓ Fetched data for {len(data)} coins")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            result['error'] = str(e)
        
        return result
    
    def _run_validation(self, symbols: List[str]) -> Dict:
        """Run data validation step."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: DATA VALIDATION")
        logger.info("=" * 60)
        
        result = {'success': True, 'validations': {}}
        
        for symbol in symbols:
            df = self.loader.load_raw_data(symbol)
            
            if df is None:
                result['validations'][symbol] = {'passed': False, 'error': 'No data'}
                continue
            
            report = self.validator.validate(df, data_type='raw')
            result['validations'][symbol] = {
                'passed': report.critical_passed,
                'results': {r.name: r.passed for r in report.results}
            }
            
            if not report.critical_passed:
                result['success'] = False
                logger.warning(f"✗ Validation issues for {symbol}")
            else:
                logger.info(f"✓ Validation passed for {symbol}")
        
        return result
    
    def _run_feature_engineering(
        self,
        symbols: List[str],
        parallel: bool = True
    ) -> Dict:
        """Run feature engineering step."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        result = {'success': True, 'features': {}}
        
        def process_symbol(symbol: str) -> tuple:
            df = self.loader.load_raw_data(symbol)
            if df is None:
                return symbol, None, "No data"
            
            try:
                featured_df = self.feature_pipeline.create_features(df)
                self.loader.save_processed_data(featured_df, symbol)
                return symbol, len(featured_df), None
            except Exception as e:
                return symbol, None, str(e)
        
        if parallel and self.config.parallel_enabled:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(process_symbol, s): s for s in symbols}
                
                for future in as_completed(futures):
                    symbol, count, error = future.result()
                    if error:
                        result['features'][symbol] = {'error': error}
                        result['success'] = False
                    else:
                        result['features'][symbol] = {'samples': count}
                        logger.info(f"✓ {symbol}: {count} samples")
        else:
            for symbol in symbols:
                symbol, count, error = process_symbol(symbol)
                if error:
                    result['features'][symbol] = {'error': error}
                    result['success'] = False
                else:
                    result['features'][symbol] = {'samples': count}
                    logger.info(f"✓ {symbol}: {count} samples")
        
        return result
    
    def _run_training(
        self,
        symbols: List[str],
        parallel: bool = True
    ) -> Dict:
        """Run model training step."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 60)
        
        result = {'success': True, 'models': {}}
        
        for symbol in symbols:
            df = self.loader.load_processed_data(symbol)
            
            if df is None:
                result['models'][symbol] = {'error': 'No processed data'}
                continue
            
            try:
                training_results = self.trainer.train_all_models(
                    df, symbol, self.feature_pipeline
                )
                
                result['models'][symbol] = {
                    key: {
                        'metrics': val.get('metrics', {}),
                        'error': val.get('error')
                    }
                    for key, val in training_results.items()
                }
                
                logger.info(f"✓ {symbol}: {len(training_results)} models trained")
                
            except Exception as e:
                result['models'][symbol] = {'error': str(e)}
                result['success'] = False
                logger.error(f"✗ Training failed for {symbol}: {e}")
        
        return result
    
    def _run_predictions(self, symbols: List[str]) -> Dict:
        """Generate predictions for all coins."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: PREDICTIONS")
        logger.info("=" * 60)
        
        from .predictor import PredictionPipeline
        
        predictor = PredictionPipeline(self.config)
        result = {'success': True, 'predictions': {}}
        
        for symbol in symbols:
            try:
                preds = predictor.predict_all_horizons(symbol)
                result['predictions'][symbol] = preds
                
                # Log summary
                for pred in preds.get('predictions', []):
                    logger.info(
                        f"  {symbol} {pred['target']}@{pred['horizon']}d: "
                        f"${pred['predicted_value']:,.2f} ({pred['change_pct']:+.2f}%)"
                    )
                
            except Exception as e:
                result['predictions'][symbol] = {'error': str(e)}
                logger.warning(f"✗ Prediction failed for {symbol}: {e}")
        
        return result
    
    def _print_summary(self, results: Dict) -> None:
        """Print execution summary."""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {results['duration_seconds']:.1f} seconds")
        logger.info(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
        
        for step_name, step_result in results.get('steps', {}).items():
            status = "✓" if step_result.get('success', False) else "✗"
            logger.info(f"  {status} {step_name}")
        
        logger.info("=" * 80)
    
    # Convenience methods for individual steps
    def ingest(self, symbols: Optional[List[str]] = None) -> Dict:
        """Run only data ingestion."""
        return self._run_ingestion(symbols or self.config.coin_symbols)
    
    def validate(self, symbols: Optional[List[str]] = None) -> Dict:
        """Run only validation."""
        return self._run_validation(symbols or self.config.coin_symbols)
    
    def engineer_features(self, symbols: Optional[List[str]] = None) -> Dict:
        """Run only feature engineering."""
        return self._run_feature_engineering(symbols or self.config.coin_symbols)
    
    def train(self, symbols: Optional[List[str]] = None) -> Dict:
        """Run only model training."""
        return self._run_training(symbols or self.config.coin_symbols)
    
    def predict(self, symbols: Optional[List[str]] = None) -> Dict:
        """Run only predictions."""
        return self._run_predictions(symbols or self.config.coin_symbols)

#!/usr/bin/env python3
"""
Main training script for the crypto forecast pipeline.

Usage:
    python scripts/train.py                    # Run full pipeline
    python scripts/train.py --step ingest      # Run specific step
    python scripts/train.py --coins BTC ETH    # Specific coins
    python scripts/train.py --skip-training    # Skip training
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.pipeline import Pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Forecast Training Pipeline"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'ingest', 'validate', 'features', 'train', 'predict'],
        default='all',
        help='Pipeline step to run'
    )
    
    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        help='Specific coins to process'
    )
    
    parser.add_argument(
        '--skip-ingestion',
        action='store_true',
        help='Skip data ingestion'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = Config(args.config)
    
    # Initialize pipeline
    pipeline = Pipeline(config)
    
    # Get coins to process
    symbols = args.coins or config.coin_symbols
    
    print(f"\n{'='*60}")
    print("CRYPTO FORECAST PIPELINE")
    print(f"{'='*60}")
    print(f"Coins: {symbols}")
    print(f"Step: {args.step}")
    print(f"{'='*60}\n")
    
    # Run requested step
    if args.step == 'all':
        results = pipeline.run(
            symbols=symbols,
            skip_ingestion=args.skip_ingestion,
            skip_training=args.skip_training,
            parallel=not args.no_parallel
        )
    elif args.step == 'ingest':
        results = pipeline.ingest(symbols)
    elif args.step == 'validate':
        results = pipeline.validate(symbols)
    elif args.step == 'features':
        results = pipeline.engineer_features(symbols)
    elif args.step == 'train':
        results = pipeline.train(symbols)
    elif args.step == 'predict':
        results = pipeline.predict(symbols)
    
    # Exit with appropriate code
    success = results.get('success', True)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

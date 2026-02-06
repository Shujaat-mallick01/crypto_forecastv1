#!/usr/bin/env python3
"""
Prediction script for generating forecasts.

Usage:
    python scripts/predict.py                  # Predict all coins
    python scripts/predict.py --coins BTC ETH  # Specific coins
    python scripts/predict.py --output out.csv # Save to file
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.pipeline import PredictionPipeline
from src.evaluation import PredictionVisualizer
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Forecast Prediction"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        help='Specific coins to predict'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not auto-save predictions'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Do not generate visualizations'
    )
    
    return parser.parse_args()


def generate_visualizations(config, predictions_df, symbols):
    """Generate and save forecast visualizations."""
    visualizer = PredictionVisualizer(config.visualizations_path)
    
    for symbol in symbols:
        symbol_preds = predictions_df[predictions_df['symbol'] == symbol]
        
        if symbol_preds.empty:
            continue
        
        # Get price predictions
        price_preds = symbol_preds[symbol_preds['target'] == 'price']
        
        if not price_preds.empty:
            current_value = price_preds.iloc[0]['current_value']
            current_date = pd.Timestamp.now()
            
            # Build predictions dict {horizon: predicted_value}
            predictions_dict = {}
            for _, row in price_preds.iterrows():
                predictions_dict[row['horizon']] = row['predicted_value']
            
            # Generate forecast timeline
            visualizer.plot_forecast_timeline(
                current_value=current_value,
                predictions=predictions_dict,
                current_date=current_date,
                symbol=symbol,
                target='price',
                save_name=f"{symbol}_price_forecast"
            )
            print(f"  - Saved {symbol} price forecast chart")
        
        # Get market cap predictions
        mcap_preds = symbol_preds[symbol_preds['target'] == 'market_cap']
        
        if not mcap_preds.empty:
            current_value = mcap_preds.iloc[0]['current_value']
            current_date = pd.Timestamp.now()
            
            predictions_dict = {}
            for _, row in mcap_preds.iterrows():
                predictions_dict[row['horizon']] = row['predicted_value']
            
            visualizer.plot_forecast_timeline(
                current_value=current_value,
                predictions=predictions_dict,
                current_date=current_date,
                symbol=symbol,
                target='market_cap',
                save_name=f"{symbol}_market_cap_forecast"
            )
            print(f"  - Saved {symbol} market cap forecast chart")
    
    # Close all figures to free memory
    import matplotlib.pyplot as plt
    plt.close('all')


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Initialize predictor
    predictor = PredictionPipeline(config)
    
    # Get coins
    symbols = args.coins or config.coin_symbols
    
    print(f"\n{'='*60}")
    print("CRYPTO FORECAST PREDICTIONS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Generate predictions
    predictions_df = predictor.predict_multiple(symbols)
    
    if predictions_df.empty:
        print("No predictions generated. Check if models are trained.")
        sys.exit(1)
    
    # Auto-save predictions unless --no-save is specified
    if not args.no_save:
        # Save predictions CSV
        saved_path = predictor.save_predictions(predictions_df)
        print(f"Predictions saved to: {saved_path}")
        
        # Generate and save report
        report = predictor.generate_report(symbols)
        print(f"Report saved to: {report.get('report_path', 'N/A')}")
    
    # Generate visualizations unless --no-viz is specified
    if not args.no_viz:
        print("\nGenerating visualizations...")
        try:
            generate_visualizations(config, predictions_df, symbols)
            print(f"Visualizations saved to: {config.visualizations_path}")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    # Output to console
    if args.format == 'table':
        print("\nFORECAST SUMMARY")
        print("-" * 80)
        
        for symbol in symbols:
            symbol_preds = predictions_df[predictions_df['symbol'] == symbol]
            
            if symbol_preds.empty:
                print(f"\n{symbol}: No predictions available")
                continue
            
            print(f"\n{symbol}:")
            
            for _, row in symbol_preds.iterrows():
                direction = "+" if row['change_pct'] > 0 else ""
                print(f"  {row['target']:12s} @{row['horizon']:2d}d: "
                      f"${row['predicted_value']:>15,.2f} ({direction}{row['change_pct']:.2f}%)")
        
        print("\n" + "-" * 80)
        
    elif args.format == 'json':
        print(predictions_df.to_json(orient='records', indent=2))
        
    elif args.format == 'csv':
        print(predictions_df.to_csv(index=False))
    
    # Save to custom file if specified
    if args.output:
        output_path = Path(args.output)
        
        if output_path.suffix == '.json':
            predictions_df.to_json(output_path, orient='records', indent=2)
        else:
            predictions_df.to_csv(output_path, index=False)
        
        print(f"\nPredictions also saved to: {output_path}")
    
    print("\n[OK] Prediction complete!")


if __name__ == '__main__':
    main()
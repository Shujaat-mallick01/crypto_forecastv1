# 🚀 Crypto Forecast

A production-ready cryptocurrency price and market cap prediction pipeline using XGBoost and LightGBM.

## Features

- **Multi-target prediction**: Price and market cap forecasts
- **Multiple horizons**: 1-day, 7-day, and 30-day predictions
- **Ensemble models**: Combines XGBoost and LightGBM
- **Technical indicators**: RSI, MACD, Bollinger Bands, and more
- **Walk-forward backtesting**: Proper time-series validation
- **Modular architecture**: Easy to extend and customize

## Project Structure

```
crypto_forecast/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── data/                 # Data ingestion and validation
│   │   ├── ingestion.py      # CoinGecko API client
│   │   ├── loader.py         # Data loading utilities
│   │   └── validator.py      # Data quality checks
│   ├── features/             # Feature engineering
│   │   ├── technical.py      # Technical indicators
│   │   ├── lag_features.py   # Lag and rolling features
│   │   └── pipeline.py       # Feature pipeline
│   ├── models/               # ML models
│   │   ├── base.py           # Base model interface
│   │   ├── xgboost_model.py  # XGBoost implementation
│   │   ├── lightgbm_model.py # LightGBM implementation
│   │   ├── ensemble.py       # Ensemble model
│   │   └── trainer.py        # Training orchestration
│   ├── evaluation/           # Model evaluation
│   │   ├── metrics.py        # Performance metrics
│   │   ├── backtester.py     # Walk-forward validation
│   │   └── visualizer.py     # Visualization tools
│   └── pipeline/             # Pipeline orchestration
│       ├── orchestrator.py   # Main pipeline
│       └── predictor.py      # Prediction pipeline
├── scripts/
│   ├── train.py              # Training script
│   └── predict.py            # Prediction script
├── data/                     # Data storage
├── models/                   # Trained models
├── logs/                     # Log files
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
# Clone or copy the project
cd crypto_forecast

# Create virtual environment (recommended)
python -m venv <venv>
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run complete pipeline (ingest → features → train → predict)
python scripts/train.py

# Or run specific steps
python scripts/train.py --step ingest      # Fetch data only
python scripts/train.py --step features    # Feature engineering only
python scripts/train.py --step train       # Training only
```

### 3. Generate Predictions

```bash
# Predict all coins
python scripts/predict.py

# Predict specific coins
python scripts/predict.py --coins BTC ETH SOL

# Save to file
python scripts/predict.py --output predictions.csv
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Target cryptocurrencies
data:
  coins:
    - symbol: "BTC"
      coingecko_id: "bitcoin"
    # Add more...

# Model settings
model:
  targets: ["price", "market_cap"]
  horizons: [1, 7, 30]
  default_algorithm: "lightgbm"  # or "xgboost"
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 1 GB | 5 GB (SSD) |

**Estimated Training Time** (10 coins, 1 year data):
- 4-core CPU: ~20-30 minutes
- 8-core CPU: ~10-15 minutes

## Usage Examples

### Python API

```python
from src.config import Config
from src.pipeline import Pipeline, PredictionPipeline

# Run full pipeline
config = Config('config/config.yaml')
pipeline = Pipeline(config)
results = pipeline.run()

# Generate predictions only
predictor = PredictionPipeline(config)
predictions = predictor.predict_all_horizons('BTC')
print(predictions)
```

### Custom Training

```python
from src.models import XGBoostModel, LightGBMModel, EnsembleModel

# Train individual model
model = LightGBMModel('my_model', {'n_estimators': 500})
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)

# Use ensemble
ensemble = EnsembleModel(
    method='weighted_average',
    weights={'xgboost': 0.4, 'lightgbm': 0.6}
)
ensemble.fit(X_train, y_train)
```

## Performance Metrics

Models are evaluated using:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Trend prediction accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Data provided by [CoinGecko](https://www.coingecko.com/)
- Built with [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/)

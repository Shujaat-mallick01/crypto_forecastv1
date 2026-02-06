# Crypto Forecast Dashboard

A comprehensive web dashboard for cryptocurrency price and market cap predictions using machine learning models.

## Features

- **User Authentication**: Secure login/logout system with JWT tokens
- **Live Market Data**: Real-time cryptocurrency market information
- **ML Predictions**: Generate predictions for price and market cap
- **Live Training**: Train models directly from the dashboard
- **WebSocket Updates**: Real-time updates for predictions and training
- **Interactive Charts**: Visualize predictions and training metrics
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env file with your configurations
```

## Running the Dashboard

1. Start the Flask application:
```bash
python run.py
```

2. Access the dashboard:
```
http://127.0.0.1:5000
```

3. Login with default credentials:
- Username: `admin`
- Password: `admin123`

## Dashboard Pages

### 1. Main Dashboard (`/dashboard`)
- Overview statistics
- Live market data table
- Quick prediction interface
- Quick training interface
- Recent predictions chart

### 2. Predictions (`/dashboard/predictions`)
- View all predictions history
- Filter by coin, type, and date
- Prediction accuracy charts
- Export predictions data

### 3. Training (`/dashboard/training`)
- Configure and start training sessions
- Live training monitor with loss charts
- Training history table
- Model comparison charts

## API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/logout` - User logout
- `GET /auth/profile` - Get user profile

### Predictions
- `POST /api/predict/<coin_symbol>` - Generate prediction
- `GET /api/predictions/latest` - Get latest predictions
- `GET /api/predictions/history/<coin_symbol>` - Get prediction history

### Training
- `POST /api/training/start` - Start training session
- `GET /api/training/status/<session_id>` - Get training status
- `GET /api/training/history` - Get training history
- `GET /api/training/metrics/<coin_symbol>` - Get model metrics

### Market Data
- `GET /api/market/live` - Get live market data
- `GET /api/coins/available` - Get available coins

## WebSocket Events

### Predictions Namespace (`/predictions`)
- `new_prediction` - Fired when a new prediction is made
- `prediction_update` - Fired when prediction status updates

### Training Namespace (`/training`)
- `training_status` - Training progress updates
- `training_complete` - Training completed successfully
- `training_failed` - Training failed with error

## Configuration

Edit the `.env` file to customize:

- `SECRET_KEY`: Flask secret key for sessions
- `JWT_SECRET_KEY`: JWT secret for token generation
- `DATABASE_URL`: Database connection string
- `FLASK_DEBUG`: Enable/disable debug mode
- `FLASK_HOST`: Host to bind the server
- `FLASK_PORT`: Port to run the server

## Security Notes

⚠️ **Important**: Change the following in production:
1. Default admin credentials
2. Secret keys in `.env` file
3. Enable HTTPS
4. Use a production database (PostgreSQL/MySQL)
5. Implement rate limiting
6. Add CSRF protection

## Project Structure

```
dashboard/
├── app/
│   ├── __init__.py           # Flask app initialization
│   ├── models/                # Database models
│   │   ├── user.py
│   │   └── prediction.py
│   ├── routes/                # API routes
│   │   ├── auth.py
│   │   ├── api.py
│   │   ├── dashboard.py
│   │   └── training.py
│   ├── services/              # Business logic
│   │   ├── ml_service.py
│   │   └── training_service.py
│   ├── static/                # Static files
│   │   ├── css/
│   │   └── js/
│   └── templates/             # HTML templates
├── requirements.txt           # Python dependencies
├── run.py                     # Application entry point
├── .env                       # Environment variables
└── README.md                  # This file
```

## Troubleshooting

### Database Issues
If you encounter database errors:
```bash
# Delete existing database
rm crypto_dashboard.db

# Restart the application
python run.py
```

### Port Already in Use
If port 5000 is already in use:
```bash
# Use a different port
FLASK_PORT=5001 python run.py
```

### WebSocket Connection Issues
- Ensure firewall allows WebSocket connections
- Check browser console for errors
- Try disabling browser extensions

## Development

### Adding New Features
1. Create new routes in `app/routes/`
2. Add corresponding templates in `app/templates/`
3. Update JavaScript in `app/static/js/`
4. Add new models if needed in `app/models/`

### Testing
```bash
# Run tests (when implemented)
pytest tests/
```

## Support

For issues or questions, please create an issue in the project repository.
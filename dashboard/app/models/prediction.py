from app import db
from datetime import datetime


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    coin_symbol = db.Column(db.String(10), nullable=False)
    prediction_type = db.Column(db.String(20), nullable=False)  # 'price' or 'market_cap'
    current_value = db.Column(db.Float, nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False)
    confidence = db.Column(db.Float)
    model_version = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'coin_symbol': self.coin_symbol,
            'prediction_type': self.prediction_type,
            'current_value': self.current_value,
            'predicted_value': self.predicted_value,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'confidence': self.confidence,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TrainingSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    coin_symbol = db.Column(db.String(10), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    metrics = db.Column(db.JSON)  # Store training metrics as JSON
    error_message = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'coin_symbol': self.coin_symbol,
            'model_type': self.model_type,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metrics': self.metrics,
            'error_message': self.error_message
        }
from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from app import db, socketio
from app.models.prediction import Prediction
from app.services.ml_service import MLService
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)
ml_service = MLService()


@api_bp.route('/predict/<coin_symbol>', methods=['POST'])
@login_required
def predict(coin_symbol):
    """Generate predictions for a specific coin."""
    try:
        data = request.get_json() or {}
        prediction_type = data.get('type', 'price')
        horizon = data.get('horizon', None)  # None = all horizons

        # Get predictions from ML service
        result = ml_service.predict(coin_symbol, prediction_type, horizon)

        if result['success']:
            predictions_list = result.get('predictions', [])
            saved_predictions = []

            for pred in predictions_list:
                # Store each prediction in database
                db_pred = Prediction(
                    user_id=current_user.id,
                    coin_symbol=coin_symbol.upper(),
                    prediction_type=pred['target'],
                    current_value=pred['current_value'],
                    predicted_value=pred['predicted_value'],
                    prediction_date=datetime.utcnow() + timedelta(days=pred['horizon']),
                    confidence=abs(100 - abs(pred['change_pct'])) / 100,  # Simple confidence from change magnitude
                    model_version=pred.get('model_version', 'ensemble')
                )
                db.session.add(db_pred)
                saved_predictions.append(pred)

            db.session.commit()

            # Emit via WebSocket
            socketio.emit('new_prediction', {
                'symbol': coin_symbol.upper(),
                'predictions': saved_predictions,
                'user_id': current_user.id
            }, namespace='/predictions')

            return jsonify({
                'success': True,
                'symbol': coin_symbol.upper(),
                'current_price': result.get('current_price'),
                'current_market_cap': result.get('current_market_cap'),
                'predictions': saved_predictions,
                'generated_at': result.get('generated_at')
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Prediction failed')
            }), 400

    except Exception as e:
        logger.error(f"Prediction API error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/predictions/latest', methods=['GET'])
@login_required
def get_latest_predictions():
    """Get latest predictions for all coins."""
    try:
        limit = request.args.get('limit', 20, type=int)

        predictions = Prediction.query.filter_by(user_id=current_user.id)\
            .order_by(Prediction.created_at.desc())\
            .limit(limit)\
            .all()

        return jsonify({
            'success': True,
            'predictions': [p.to_dict() for p in predictions]
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/predictions/history/<coin_symbol>', methods=['GET'])
@login_required
def get_prediction_history(coin_symbol):
    """Get prediction history for a specific coin."""
    try:
        days = request.args.get('days', 30, type=int)
        since_date = datetime.utcnow() - timedelta(days=days)

        predictions = Prediction.query.filter_by(
            user_id=current_user.id,
            coin_symbol=coin_symbol.upper()
        ).filter(
            Prediction.created_at >= since_date
        ).order_by(Prediction.created_at.desc()).all()

        return jsonify({
            'success': True,
            'coin': coin_symbol.upper(),
            'predictions': [p.to_dict() for p in predictions]
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/coins/available', methods=['GET'])
@login_required
def get_available_coins():
    """Get list of available coins."""
    try:
        coins = ml_service.get_available_coins()
        coins_with_models = ml_service.get_coins_with_models()
        coins_with_data = ml_service.get_coins_with_data()

        return jsonify({
            'success': True,
            'coins': coins,
            'coins_with_models': coins_with_models,
            'coins_with_data': coins_with_data
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/market/live', methods=['GET'])
@login_required
def get_live_market_data():
    """Get market data for all coins."""
    try:
        market_data = ml_service.get_live_market_data()
        return jsonify({
            'success': True,
            'data': market_data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/models/available', methods=['GET'])
@login_required
def get_available_models():
    """Get list of all trained models."""
    try:
        models = ml_service.get_available_models()
        return jsonify({
            'success': True,
            'models': models
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

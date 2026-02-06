from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
from app.models.prediction import Prediction, TrainingSession
from app.services.ml_service import MLService
from app import db

dashboard_bp = Blueprint('dashboard', __name__)
ml_service = MLService()


@dashboard_bp.route('/')
@login_required
def index():
    coins = ml_service.get_available_coins()
    coins_with_models = ml_service.get_coins_with_models()
    return render_template(
        'dashboard.html',
        user=current_user,
        coins=coins,
        coins_with_models=coins_with_models
    )


@dashboard_bp.route('/predictions')
@login_required
def predictions():
    user_predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.created_at.desc()).limit(100).all()

    coins = ml_service.get_available_coins()
    return render_template(
        'predictions.html',
        predictions=user_predictions,
        coins=coins
    )


@dashboard_bp.route('/training')
@login_required
def training():
    sessions = TrainingSession.query.filter_by(
        user_id=current_user.id
    ).order_by(TrainingSession.start_time.desc()).all()

    coins = ml_service.get_available_coins()
    coins_with_models = ml_service.get_coins_with_models()
    return render_template(
        'training.html',
        sessions=sessions,
        coins=coins,
        coins_with_models=coins_with_models
    )


@dashboard_bp.route('/stats')
@login_required
def stats():
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    total_training_sessions = TrainingSession.query.filter_by(user_id=current_user.id).count()
    successful_sessions = TrainingSession.query.filter_by(user_id=current_user.id, status='completed').count()
    coins_with_models = ml_service.get_coins_with_models()

    stats_data = {
        'total_predictions': total_predictions,
        'total_training_sessions': total_training_sessions,
        'successful_sessions': successful_sessions,
        'success_rate': (successful_sessions / total_training_sessions * 100) if total_training_sessions > 0 else 0,
        'coins_with_models': len(coins_with_models),
        'total_coins': len(ml_service.get_available_coins())
    }

    return jsonify(stats_data)

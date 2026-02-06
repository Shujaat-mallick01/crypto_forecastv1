from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from app import db, socketio
from app.models.prediction import TrainingSession
from app.services.training_service import TrainingService
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)

training_bp = Blueprint('training', __name__)
training_service = TrainingService()


@training_bp.route('/start', methods=['POST'])
@login_required
def start_training():
    """Start a new training session."""
    try:
        data = request.get_json()

        coin_symbols = data.get('coin_symbols', [])
        if not coin_symbols:
            # Support single coin for backwards compat
            single = data.get('coin_symbol')
            if single:
                coin_symbols = [single]

        if not coin_symbols:
            return jsonify({'error': 'At least one coin symbol is required'}), 400

        model_type = data.get('model_type', 'ensemble')
        skip_ingestion = data.get('skip_ingestion', False)

        # Normalize
        coin_symbols = [s.upper() for s in coin_symbols]

        # Create training session
        session = TrainingSession(
            user_id=current_user.id,
            coin_symbol=','.join(coin_symbols),
            model_type=model_type,
            status='pending'
        )
        db.session.add(session)
        db.session.commit()

        # Get app for context in thread
        app = current_app._get_current_object()

        # Start training in background thread
        thread = threading.Thread(
            target=run_training,
            args=(app, session.id, current_user.id, coin_symbols, model_type, skip_ingestion)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'session_id': session.id,
            'message': f'Training started for {", ".join(coin_symbols)}'
        }), 200

    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def run_training(app, session_id, user_id, coin_symbols, model_type, skip_ingestion):
    """Run training in background with app context."""
    with app.app_context():
        try:
            # Update session status
            session = db.session.get(TrainingSession, session_id)
            session.status = 'running'
            db.session.commit()

            # Emit status update
            socketio.emit('training_status', {
                'session_id': session_id,
                'status': 'running',
                'message': f'Training {model_type} models for {", ".join(coin_symbols)}...',
                'progress': 10
            }, namespace='/training')

            # Run actual training
            result = training_service.train_model(
                coin_symbols, model_type, skip_ingestion
            )

            # Update session with results
            session = db.session.get(TrainingSession, session_id)
            if result['success']:
                session.status = 'completed'
                session.metrics = result.get('metrics', {})
                session.end_time = datetime.utcnow()

                socketio.emit('training_complete', {
                    'session_id': session_id,
                    'status': 'completed',
                    'metrics': session.metrics,
                    'duration': result.get('duration', 0)
                }, namespace='/training')
            else:
                session.status = 'failed'
                session.error_message = result.get('error', 'Unknown error')
                session.end_time = datetime.utcnow()

                socketio.emit('training_failed', {
                    'session_id': session_id,
                    'status': 'failed',
                    'error': session.error_message
                }, namespace='/training')

            db.session.commit()

        except Exception as e:
            logger.error(f"Training thread error: {e}", exc_info=True)
            try:
                session = db.session.get(TrainingSession, session_id)
                session.status = 'failed'
                session.error_message = str(e)
                session.end_time = datetime.utcnow()
                db.session.commit()
            except Exception:
                pass

            socketio.emit('training_failed', {
                'session_id': session_id,
                'status': 'failed',
                'error': str(e)
            }, namespace='/training')


@training_bp.route('/status/<int:session_id>', methods=['GET'])
@login_required
def get_training_status(session_id):
    """Get status of a training session."""
    try:
        session = TrainingSession.query.filter_by(
            id=session_id,
            user_id=current_user.id
        ).first()

        if not session:
            return jsonify({'error': 'Session not found'}), 404

        return jsonify({
            'success': True,
            'session': session.to_dict()
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/history', methods=['GET'])
@login_required
def get_training_history():
    """Get training history for the current user."""
    try:
        limit = request.args.get('limit', 20, type=int)

        sessions = TrainingSession.query.filter_by(user_id=current_user.id)\
            .order_by(TrainingSession.start_time.desc())\
            .limit(limit)\
            .all()

        return jsonify({
            'success': True,
            'sessions': [s.to_dict() for s in sessions]
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/metrics/<coin_symbol>', methods=['GET'])
@login_required
def get_model_metrics(coin_symbol):
    """Get metrics for a specific coin's models."""
    try:
        metrics = training_service.get_model_metrics(coin_symbol.upper())

        return jsonify({
            'success': True,
            'coin': coin_symbol.upper(),
            'metrics': metrics
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@training_bp.route('/summary', methods=['GET'])
@login_required
def get_training_summary():
    """Get summary of all available models and data."""
    try:
        summary = training_service.get_training_status_summary()
        return jsonify({
            'success': True,
            'summary': summary
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

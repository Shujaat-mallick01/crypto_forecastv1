#!/usr/bin/env python3
"""
Run the Crypto Forecast Dashboard application.

Usage:
    python dashboard/run.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so src package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add dashboard directory so app package is importable
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app, socketio

# Create Flask app
app = create_app()

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    print(f"""
    ============================================================
             CRYPTO FORECAST DASHBOARD

      Access the dashboard at:
      http://{host}:{port}

      Default credentials:
      Username: admin
      Password: admin123

      Press Ctrl+C to stop the server
    ============================================================
    """)

    # Run with SocketIO (threading mode, no eventlet needed)
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

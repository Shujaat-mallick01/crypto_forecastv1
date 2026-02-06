// Main JavaScript file for Crypto Forecast Dashboard

let socket = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeWebSocket();
});

// Initialize WebSocket connection
function initializeWebSocket() {
    try {
        socket = io('/predictions', {
            transports: ['websocket', 'polling']
        });

        socket.on('connect', function() {
            console.log('WebSocket connected');
            updateConnectionStatus('connected');
        });

        socket.on('disconnect', function() {
            console.log('WebSocket disconnected');
            updateConnectionStatus('disconnected');
        });

        socket.on('new_prediction', function(data) {
            console.log('New prediction received:', data);
            if (typeof handleNewPrediction === 'function') {
                handleNewPrediction(data);
            }
        });

        socket.on('training_status', function(data) {
            console.log('Training status:', data);
            if (typeof handleTrainingUpdate === 'function') {
                handleTrainingUpdate(data);
            }
        });
    } catch (e) {
        console.log('WebSocket init skipped:', e.message);
    }
}

// Update connection status indicator
function updateConnectionStatus(status) {
    const el = document.getElementById('websocket-status');
    if (el) {
        el.className = 'badge bg-' + (status === 'connected' ? 'success' : 'secondary');
        el.innerHTML = '<i class="fas fa-circle"></i> ' +
            (status === 'connected' ? 'Connected' : 'Disconnected');
    }
}

// API helper - uses session cookies, no JWT
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        },
        credentials: 'same-origin'
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    // Prepend /api if not already present
    const url = endpoint.startsWith('/api') ? endpoint : '/api' + endpoint;

    const response = await fetch(url, options);
    const responseData = await response.json();

    if (!response.ok) {
        throw new Error(responseData.error || `API call failed (${response.status})`);
    }

    return responseData;
}

// Training API helper
async function trainingApiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        },
        credentials: 'same-origin'
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    const url = '/training' + endpoint;
    const response = await fetch(url, options);
    const responseData = await response.json();

    if (!response.ok) {
        throw new Error(responseData.error || `API call failed (${response.status})`);
    }

    return responseData;
}

// Show notification toast
function showNotification(type, message) {
    const alertClass = type === 'success' ? 'alert-success' :
                       type === 'error' ? 'alert-danger' :
                       type === 'warning' ? 'alert-warning' : 'alert-info';

    const alert = document.createElement('div');
    alert.className = `alert ${alertClass} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    alert.style.zIndex = '9999';
    alert.style.maxWidth = '400px';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alert);

    setTimeout(() => {
        if (alert.parentNode) alert.remove();
    }, 5000);
}

// Format currency
function formatCurrency(value) {
    if (value === 0 || value === null || value === undefined) return '$0.00';
    if (Math.abs(value) >= 1e9) {
        return '$' + (value / 1e9).toFixed(2) + 'B';
    }
    if (Math.abs(value) >= 1e6) {
        return '$' + (value / 1e6).toFixed(2) + 'M';
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

// Format percentage
function formatPercentage(value) {
    const formatted = (value * 100).toFixed(2);
    return `${value >= 0 ? '+' : ''}${formatted}%`;
}

// Format date
function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Export functions
window.apiCall = apiCall;
window.trainingApiCall = trainingApiCall;
window.showNotification = showNotification;
window.formatCurrency = formatCurrency;
window.formatPercentage = formatPercentage;
window.formatDate = formatDate;

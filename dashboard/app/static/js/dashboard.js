// Dashboard JavaScript

let marketDataInterval = null;

document.addEventListener('DOMContentLoaded', function() {
    loadDashboardStats();
    loadMarketData();
    loadRecentPredictions();
    setupDashboardEventListeners();

    // Refresh market data every 60 seconds
    marketDataInterval = setInterval(loadMarketData, 60000);
});

// Load dashboard statistics
async function loadDashboardStats() {
    try {
        const stats = await apiCall('/dashboard/stats');
        document.getElementById('total-predictions').textContent = stats.total_predictions || 0;
        document.getElementById('models-trained').textContent = stats.total_training_sessions || 0;
        document.getElementById('coins-with-models').textContent = stats.coins_with_models || 0;
        document.getElementById('success-rate').textContent = (stats.success_rate || 0).toFixed(1) + '%';
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load market data
async function loadMarketData() {
    try {
        const response = await apiCall('/market/live');
        const marketData = response.data;

        const tableBody = document.getElementById('market-data');
        tableBody.innerHTML = '';

        let hasData = false;
        for (const [coin, data] of Object.entries(marketData)) {
            const row = document.createElement('tr');
            const noData = data.no_data || data.price === 0;

            if (!noData) hasData = true;

            const changeClass = data.change_24h >= 0 ? 'text-success' : 'text-danger';
            const changeIcon = data.change_24h >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';

            row.innerHTML = `
                <td><strong>${coin}</strong></td>
                <td>${noData ? '<span class="text-muted">No data</span>' : formatCurrency(data.price)}</td>
                <td>${noData ? '-' : formatCurrency(data.market_cap)}</td>
                <td class="${changeClass}">
                    ${noData ? '-' : `<i class="fas ${changeIcon}"></i> ${data.change_24h.toFixed(2)}%`}
                </td>
                <td>${noData ? '-' : formatCurrency(data.volume)}</td>
                <td>
                    ${noData
                        ? '<span class="badge bg-secondary">No Data</span>'
                        : '<span class="badge bg-success">Ready</span>'}
                </td>
                <td>
                    ${noData
                        ? ''
                        : `<button class="btn btn-sm btn-primary" onclick="quickPredict('${coin}')" title="Predict">
                             <i class="fas fa-chart-line"></i>
                           </button>`}
                </td>
            `;
            tableBody.appendChild(row);
        }

        document.getElementById('data-status').textContent = hasData ? 'Data Available' : 'No Data - Train First';
        document.getElementById('data-status').className = 'badge bg-' + (hasData ? 'success' : 'warning');

    } catch (error) {
        console.error('Error loading market data:', error);
        document.getElementById('data-status').textContent = 'Error';
        document.getElementById('data-status').className = 'badge bg-danger';
    }
}

// Load recent predictions
async function loadRecentPredictions() {
    try {
        const response = await apiCall('/predictions/latest?limit=10');
        const predictions = response.predictions;

        const tbody = document.getElementById('recent-predictions');
        if (!predictions || predictions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No predictions yet. Train a model and generate predictions!</td></tr>';
            return;
        }

        tbody.innerHTML = '';
        predictions.forEach(p => {
            const change = ((p.predicted_value - p.current_value) / p.current_value * 100);
            const changeClass = change >= 0 ? 'success' : 'danger';
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${formatDate(p.created_at)}</td>
                <td><span class="badge bg-primary">${p.coin_symbol}</span></td>
                <td>${p.prediction_type}</td>
                <td>${p.prediction_date ? new Date(p.prediction_date).toLocaleDateString() : '-'}</td>
                <td>${formatCurrency(p.current_value)}</td>
                <td>${formatCurrency(p.predicted_value)}</td>
                <td><span class="badge bg-${changeClass}">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</span></td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading recent predictions:', error);
    }
}

// Setup event listeners
function setupDashboardEventListeners() {
    // Prediction form
    const predForm = document.getElementById('prediction-form');
    if (predForm) {
        predForm.addEventListener('submit', handlePrediction);
    }

    // Training form
    const trainForm = document.getElementById('training-form');
    if (trainForm) {
        trainForm.addEventListener('submit', handleTraining);
    }
}

// Handle prediction
async function handlePrediction(event) {
    event.preventDefault();

    const coin = document.getElementById('coin-select').value;
    const type = document.getElementById('prediction-type').value;

    if (!coin) {
        showNotification('warning', 'Please select a coin');
        return;
    }

    const btn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('prediction-result');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Generating...';
    resultDiv.innerHTML = '<div class="text-center"><div class="spinner-border text-primary"></div><p class="mt-2">Running prediction pipeline...</p></div>';

    try {
        const response = await apiCall(`/predict/${coin}`, 'POST', { type });
        const predictions = response.predictions;

        if (!predictions || predictions.length === 0) {
            resultDiv.innerHTML = '<div class="alert alert-warning">No predictions generated. Make sure models are trained for this coin.</div>';
            return;
        }

        let html = '<div class="alert alert-success"><h5>Predictions for ' + coin + '</h5>';
        html += '<table class="table table-sm table-borderless mb-0">';
        html += '<thead><tr><th>Target</th><th>Horizon</th><th>Current</th><th>Predicted</th><th>Change</th></tr></thead><tbody>';

        predictions.forEach(p => {
            const changeClass = p.change_pct >= 0 ? 'success' : 'danger';
            const arrow = p.change_pct >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
            html += `<tr>
                <td>${p.target}</td>
                <td>${p.horizon} day${p.horizon > 1 ? 's' : ''}</td>
                <td>${formatCurrency(p.current_value)}</td>
                <td>${formatCurrency(p.predicted_value)}</td>
                <td><span class="badge bg-${changeClass}"><i class="fas ${arrow}"></i> ${p.change_pct >= 0 ? '+' : ''}${p.change_pct.toFixed(2)}%</span></td>
            </tr>`;
        });

        html += '</tbody></table></div>';
        resultDiv.innerHTML = html;

        // Refresh recent predictions
        loadRecentPredictions();
        loadDashboardStats();

    } catch (error) {
        resultDiv.innerHTML = `<div class="alert alert-danger"><strong>Error:</strong> ${error.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-chart-line"></i> Generate Prediction';
    }
}

// Handle training
async function handleTraining(event) {
    event.preventDefault();

    const selectEl = document.getElementById('train-coin-select');
    const selectedCoins = Array.from(selectEl.selectedOptions).map(o => o.value);
    const modelType = document.getElementById('model-type').value;
    const skipIngestion = document.getElementById('skip-ingestion').checked;

    if (selectedCoins.length === 0) {
        showNotification('warning', 'Please select at least one coin');
        return;
    }

    const btn = document.getElementById('train-btn');
    const statusDiv = document.getElementById('training-status');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting...';

    statusDiv.innerHTML = `
        <div class="alert alert-info">
            <h6>Training Pipeline Started</h6>
            <p>Coins: ${selectedCoins.join(', ')}</p>
            <p>Model: ${modelType}</p>
            <div class="progress mt-2">
                <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%">
                    Processing...
                </div>
            </div>
            <small class="text-muted">This may take several minutes. You can check the Training page for details.</small>
        </div>
    `;

    try {
        const response = await trainingApiCall('/start', 'POST', {
            coin_symbols: selectedCoins,
            model_type: modelType,
            skip_ingestion: skipIngestion
        });

        statusDiv.innerHTML = `
            <div class="alert alert-info">
                <h6>Training In Progress</h6>
                <p>Session ID: ${response.session_id}</p>
                <p>${response.message}</p>
                <div class="progress mt-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%">
                        Running...
                    </div>
                </div>
                <small>Check <a href="/dashboard/training">Training page</a> for live updates.</small>
            </div>
        `;

        showNotification('success', 'Training started successfully!');

        // Poll for status
        pollTrainingStatus(response.session_id, statusDiv);

    } catch (error) {
        statusDiv.innerHTML = `<div class="alert alert-danger"><strong>Error:</strong> ${error.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Start Training Pipeline';
    }
}

// Poll training status
async function pollTrainingStatus(sessionId, statusDiv) {
    const poll = async () => {
        try {
            const response = await trainingApiCall(`/status/${sessionId}`);
            const session = response.session;

            if (session.status === 'completed') {
                statusDiv.innerHTML = `
                    <div class="alert alert-success">
                        <h6>Training Complete!</h6>
                        <p>Session: ${sessionId}</p>
                        ${session.metrics ? '<p>Metrics: ' + JSON.stringify(session.metrics).substring(0, 200) + '...</p>' : ''}
                        <p><a href="/dashboard/training" class="btn btn-sm btn-outline-success">View Details</a></p>
                    </div>
                `;
                showNotification('success', 'Training completed! Models are ready for predictions.');
                loadDashboardStats();
                loadMarketData();
                return;
            }

            if (session.status === 'failed') {
                statusDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <h6>Training Failed</h6>
                        <p>${session.error_message || 'Unknown error'}</p>
                    </div>
                `;
                showNotification('error', 'Training failed: ' + (session.error_message || 'Unknown error'));
                return;
            }

            // Still running, continue polling
            setTimeout(poll, 5000);

        } catch (error) {
            console.error('Error polling status:', error);
            setTimeout(poll, 10000);
        }
    };

    setTimeout(poll, 5000);
}

// Quick predict from market table
function quickPredict(coin) {
    document.getElementById('coin-select').value = coin;
    document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
}

// Handle new prediction from WebSocket
function handleNewPrediction(data) {
    loadRecentPredictions();
    loadDashboardStats();
    showNotification('info', `New prediction for ${data.symbol}`);
}

// Cleanup
window.addEventListener('beforeunload', function() {
    if (marketDataInterval) clearInterval(marketDataInterval);
});

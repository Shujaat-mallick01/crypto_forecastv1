// Training page JavaScript

let trainingSocket = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeTrainingSocket();
    setupTrainingForm();
    loadModelStatus();
});

// Initialize training WebSocket
function initializeTrainingSocket() {
    try {
        trainingSocket = io('/training', {
            transports: ['websocket', 'polling']
        });

        trainingSocket.on('training_status', function(data) {
            updateTrainingMonitor(data);
        });

        trainingSocket.on('training_complete', function(data) {
            onTrainingComplete(data);
        });

        trainingSocket.on('training_failed', function(data) {
            onTrainingFailed(data);
        });
    } catch (e) {
        console.log('Training socket init skipped:', e.message);
    }
}

// Setup training form
function setupTrainingForm() {
    const form = document.getElementById('training-config-form');
    if (form) {
        form.addEventListener('submit', startTraining);
    }

    const selectAllBtn = document.getElementById('select-all-coins');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', function() {
            const select = document.getElementById('config-coins');
            for (let option of select.options) {
                option.selected = true;
            }
        });
    }
}

// Load model status
async function loadModelStatus() {
    try {
        const response = await trainingApiCall('/summary');
        const summary = response.summary;
        const container = document.getElementById('model-status-content');

        if (!summary) {
            container.innerHTML = '<div class="text-muted">Could not load model status</div>';
            return;
        }

        let html = '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr><th>Coin</th><th>Raw Data</th><th>Processed</th><th>Models</th></tr></thead><tbody>';

        for (const coin of summary.coins_configured) {
            const data = summary.data_available[coin] || {};
            const models = summary.models_available[coin] || {};
            const modelCount = Object.values(models).filter(v => v).length;
            const totalModels = Object.keys(models).length;

            html += `<tr>
                <td><strong>${coin}</strong></td>
                <td>${data.raw
                    ? `<span class="badge bg-success">${data.raw_rows} rows</span>`
                    : '<span class="badge bg-secondary">None</span>'}
                </td>
                <td>${data.processed
                    ? `<span class="badge bg-success">${data.processed_rows} rows</span>`
                    : '<span class="badge bg-secondary">None</span>'}
                </td>
                <td>${modelCount > 0
                    ? `<span class="badge bg-success">${modelCount}/${totalModels}</span>`
                    : '<span class="badge bg-secondary">None</span>'}
                </td>
            </tr>`;
        }

        html += '</tbody></table></div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Error loading model status:', error);
        document.getElementById('model-status-content').innerHTML =
            '<div class="text-muted">Error loading model status</div>';
    }
}

// Start training
async function startTraining(event) {
    event.preventDefault();

    const selectEl = document.getElementById('config-coins');
    const selectedCoins = Array.from(selectEl.selectedOptions).map(o => o.value);
    const modelType = document.getElementById('config-model').value;
    const skipIngestion = document.getElementById('config-skip-ingestion').checked;

    if (selectedCoins.length === 0) {
        showNotification('warning', 'Please select at least one coin');
        return;
    }

    const btn = document.getElementById('start-training-btn');
    const monitor = document.getElementById('training-monitor');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Starting...';

    monitor.innerHTML = `
        <div class="alert alert-info">
            <h6><span class="spinner-border spinner-border-sm"></span> Initializing Training Pipeline</h6>
            <p><strong>Coins:</strong> ${selectedCoins.join(', ')}</p>
            <p><strong>Model:</strong> ${modelType}</p>
            <p><strong>Data Download:</strong> ${skipIngestion ? 'Skipped' : 'Enabled'}</p>
            <div class="progress mt-2">
                <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%">
                    Starting pipeline...
                </div>
            </div>
        </div>
    `;

    try {
        const response = await trainingApiCall('/start', 'POST', {
            coin_symbols: selectedCoins,
            model_type: modelType,
            skip_ingestion: skipIngestion
        });

        monitor.innerHTML = `
            <div class="alert alert-info">
                <h6><span class="spinner-border spinner-border-sm"></span> Training In Progress</h6>
                <p>Session ID: <strong>${response.session_id}</strong></p>
                <p>${response.message}</p>
                <div class="progress mt-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated bg-success" style="width: 100%" id="training-progress">
                        Running pipeline...
                    </div>
                </div>
                <div class="mt-2" id="training-steps">
                    <small class="text-muted">Steps: Data Ingestion > Validation > Features > Training > Predictions</small>
                </div>
            </div>
        `;

        showNotification('success', 'Training started!');

        // Poll for completion
        pollTrainingStatus(response.session_id);

    } catch (error) {
        monitor.innerHTML = `<div class="alert alert-danger"><strong>Error:</strong> ${error.message}</div>`;
        showNotification('error', 'Failed to start training: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-rocket"></i> Start Full Pipeline';
    }
}

// Poll training status
async function pollTrainingStatus(sessionId) {
    const poll = async () => {
        try {
            const response = await trainingApiCall(`/status/${sessionId}`);
            const session = response.session;

            if (session.status === 'completed') {
                onTrainingComplete({
                    session_id: sessionId,
                    metrics: session.metrics,
                    status: 'completed'
                });
                return;
            }

            if (session.status === 'failed') {
                onTrainingFailed({
                    session_id: sessionId,
                    error: session.error_message,
                    status: 'failed'
                });
                return;
            }

            // Still running
            setTimeout(poll, 5000);
        } catch (error) {
            console.error('Poll error:', error);
            setTimeout(poll, 10000);
        }
    };

    setTimeout(poll, 5000);
}

// Update training monitor
function updateTrainingMonitor(data) {
    const progressBar = document.getElementById('training-progress');
    if (progressBar && data.progress) {
        progressBar.style.width = data.progress + '%';
        progressBar.textContent = data.message || (data.progress + '%');
    }
}

// On training complete
function onTrainingComplete(data) {
    const monitor = document.getElementById('training-monitor');

    let metricsHtml = '';
    if (data.metrics && typeof data.metrics === 'object') {
        metricsHtml = '<h6>Metrics Summary:</h6><div class="table-responsive"><table class="table table-sm">';
        metricsHtml += '<thead><tr><th>Coin</th><th>Model</th><th>Metric</th><th>Value</th></tr></thead><tbody>';

        for (const [coin, models] of Object.entries(data.metrics)) {
            for (const [model, metrics] of Object.entries(models)) {
                for (const [metric, value] of Object.entries(metrics)) {
                    metricsHtml += `<tr>
                        <td>${coin}</td>
                        <td>${model}</td>
                        <td>${metric}</td>
                        <td>${typeof value === 'number' ? value.toFixed(4) : value}</td>
                    </tr>`;
                }
            }
        }
        metricsHtml += '</tbody></table></div>';
    }

    monitor.innerHTML = `
        <div class="alert alert-success">
            <h6><i class="fas fa-check-circle"></i> Training Complete!</h6>
            <p>Session ID: ${data.session_id}</p>
            ${data.duration ? `<p>Duration: ${(data.duration / 60).toFixed(1)} minutes</p>` : ''}
            ${metricsHtml}
        </div>
    `;

    showNotification('success', 'Training completed! Models are ready for predictions.');

    // Refresh model status and page
    loadModelStatus();
    setTimeout(() => location.reload(), 5000);
}

// On training failed
function onTrainingFailed(data) {
    const monitor = document.getElementById('training-monitor');

    monitor.innerHTML = `
        <div class="alert alert-danger">
            <h6><i class="fas fa-times-circle"></i> Training Failed</h6>
            <p>Session ID: ${data.session_id}</p>
            <p><strong>Error:</strong> ${data.error || 'Unknown error'}</p>
        </div>
    `;

    showNotification('error', 'Training failed: ' + (data.error || 'Unknown error'));
}

// Show metrics modal
function showMetrics(metrics) {
    const modal = new bootstrap.Modal(document.getElementById('metricsModal'));
    const content = document.getElementById('metrics-content');

    if (typeof metrics === 'string') {
        try { metrics = JSON.parse(metrics); } catch(e) {}
    }

    let html = '<div class="table-responsive"><table class="table">';

    if (typeof metrics === 'object' && metrics !== null) {
        const renderMetrics = (obj, prefix = '') => {
            for (const [key, value] of Object.entries(obj)) {
                if (typeof value === 'object' && value !== null) {
                    html += `<tr><th colspan="2" class="bg-light">${prefix}${key}</th></tr>`;
                    renderMetrics(value, '');
                } else {
                    html += `<tr><td>${key}</td><td>${typeof value === 'number' ? value.toFixed(6) : value}</td></tr>`;
                }
            }
        };
        renderMetrics(metrics);
    } else {
        html += `<tr><td>${JSON.stringify(metrics)}</td></tr>`;
    }

    html += '</table></div>';
    content.innerHTML = html;
    modal.show();
}

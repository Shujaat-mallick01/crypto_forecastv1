// Predictions page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Page is server-rendered, JS handles filtering and new predictions
});

// Generate prediction from the quick predict form
async function generatePrediction() {
    const coin = document.getElementById('pred-coin').value;
    const type = document.getElementById('pred-type').value;

    if (!coin) {
        showNotification('warning', 'Please select a coin');
        return;
    }

    const btn = document.getElementById('generate-pred-btn');
    const resultDiv = document.getElementById('new-prediction-result');
    const statusDiv = document.getElementById('pred-status');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Generating...';
    statusDiv.innerHTML = '<small class="text-muted">Running prediction pipeline...</small>';

    try {
        const response = await apiCall(`/predict/${coin}`, 'POST', { type });
        const predictions = response.predictions;

        if (!predictions || predictions.length === 0) {
            resultDiv.innerHTML = '<div class="alert alert-warning">No predictions generated. Make sure models are trained for this coin.</div>';
            statusDiv.innerHTML = '';
            return;
        }

        let html = '<div class="alert alert-success"><h6>New Predictions for ' + coin + '</h6>';
        html += '<table class="table table-sm mb-0"><thead><tr><th>Target</th><th>Horizon</th><th>Current</th><th>Predicted</th><th>Change</th></tr></thead><tbody>';

        predictions.forEach(p => {
            const cls = p.change_pct >= 0 ? 'success' : 'danger';
            const arrow = p.change_pct >= 0 ? 'up' : 'down';
            html += `<tr>
                <td>${p.target}</td>
                <td>${p.horizon}d</td>
                <td>${formatCurrency(p.current_value)}</td>
                <td>${formatCurrency(p.predicted_value)}</td>
                <td><span class="badge bg-${cls}"><i class="fas fa-arrow-${arrow}"></i> ${p.change_pct >= 0 ? '+' : ''}${p.change_pct.toFixed(2)}%</span></td>
            </tr>`;
        });

        html += '</tbody></table>';
        html += '<small class="text-muted">Refresh the page to see these in the history table.</small></div>';
        resultDiv.innerHTML = html;
        statusDiv.innerHTML = '<small class="text-success">Done!</small>';

    } catch (error) {
        resultDiv.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
        statusDiv.innerHTML = '';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-chart-line"></i> Generate';
    }
}

// Filter predictions (client-side)
function filterPredictions() {
    const coinFilter = document.getElementById('coin-filter').value;
    const typeFilter = document.getElementById('type-filter').value;

    const rows = document.querySelectorAll('.prediction-row');
    let visibleCount = 0;

    rows.forEach(row => {
        const coin = row.getAttribute('data-coin');
        const type = row.getAttribute('data-type');

        const coinMatch = !coinFilter || coin === coinFilter;
        const typeMatch = !typeFilter || type === typeFilter;

        if (coinMatch && typeMatch) {
            row.style.display = '';
            visibleCount++;
        } else {
            row.style.display = 'none';
        }
    });

    showNotification('info', `Showing ${visibleCount} predictions`);
}

// Reset filters
function resetFilters() {
    document.getElementById('coin-filter').value = '';
    document.getElementById('type-filter').value = '';

    document.querySelectorAll('.prediction-row').forEach(row => {
        row.style.display = '';
    });
}

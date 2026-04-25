// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global variables
let batchRowCount = 0;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Set today's date as default
    document.getElementById('date').valueAsDate = new Date();
    
    // Load initial data
    loadModelInfo();
    checkApiHealth();
    
    // Add event listeners
    document.getElementById('single-prediction-form').addEventListener('submit', handleSinglePrediction);
    
    // Add initial batch row
    addBatchRow();
});

// API Helper Functions
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Health Check
async function checkApiHealth() {
    try {
        const health = await apiCall('/health');
        updateApiStatus(health.status === 'healthy');
        return health;
    } catch (error) {
        updateApiStatus(false);
        showNotification('API is not accessible. Please ensure the server is running.', 'danger');
        return null;
    }
}

function updateApiStatus(isHealthy) {
    const statusElement = document.getElementById('api-status');
    if (isHealthy) {
        statusElement.className = 'badge bg-success me-3';
        statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>API Connected';
    } else {
        statusElement.className = 'badge bg-danger me-3';
        statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>API Disconnected';
    }
}

// Model Information
async function loadModelInfo() {
    try {
        const modelInfo = await apiCall('/model/info');
        
        document.getElementById('model-name').textContent = modelInfo.model_name;
        document.getElementById('model-version').textContent = modelInfo.model_version;
        document.getElementById('features-count').textContent = modelInfo.features_count;
        document.getElementById('last-updated').textContent = new Date(modelInfo.last_updated).toLocaleString();
        
        // Performance metrics
        if (modelInfo.performance_metrics) {
            document.getElementById('rmse').textContent = modelInfo.performance_metrics.rmse?.toFixed(2) || '-';
            document.getElementById('mae').textContent = modelInfo.performance_metrics.mae?.toFixed(2) || '-';
            document.getElementById('r2').textContent = modelInfo.performance_metrics.r2?.toFixed(3) || '-';
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
        showNotification('Failed to load model information', 'warning');
    }
}

// Single Prediction
async function handleSinglePrediction(event) {
    event.preventDefault();
    
    const formData = {
        product_id: document.getElementById('product-id').value,
        date: document.getElementById('date').value,
        region: document.getElementById('region').value,
        price: parseFloat(document.getElementById('price').value) || null,
        category: document.getElementById('category').value || null
    };
    
    showLoading('Making prediction...');
    
    try {
        const result = await apiCall('/predict-demand', 'POST', formData);
        displaySingleResult(result);
        showNotification('Prediction completed successfully!', 'success');
    } catch (error) {
        showNotification('Prediction failed: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

function displaySingleResult(result) {
    document.getElementById('predicted-demand').textContent = result.predicted_demand;
    document.getElementById('result-product-id').textContent = result.product_id;
    document.getElementById('result-date').textContent = result.date;
    document.getElementById('result-region').textContent = result.region;
    document.getElementById('result-model-version').textContent = result.model_version;
    document.getElementById('result-timestamp').textContent = new Date(result.prediction_timestamp).toLocaleString();
    
    // Confidence interval
    if (result.confidence_interval) {
        const lower = result.confidence_interval.lower;
        const upper = result.confidence_interval.upper;
        const prediction = result.predicted_demand;
        
        document.getElementById('ci-lower').textContent = lower.toFixed(2);
        document.getElementById('ci-upper').textContent = upper.toFixed(2);
        
        // Update progress bars
        const maxVal = Math.max(upper, prediction * 1.5);
        const lowerPercent = (lower / maxVal) * 100;
        const predictionPercent = (prediction / maxVal) * 100;
        const upperPercent = ((upper - prediction) / maxVal) * 100;
        
        document.getElementById('confidence-lower').style.width = lowerPercent + '%';
        document.getElementById('confidence-prediction').style.width = predictionPercent + '%';
        document.getElementById('confidence-prediction').textContent = prediction.toFixed(0);
        document.getElementById('confidence-upper').style.width = upperPercent + '%';
        document.getElementById('confidence-upper').textContent = upper.toFixed(0);
    }
    
    document.getElementById('single-result').style.display = 'block';
    
    // Scroll to result
    document.getElementById('single-result').scrollIntoView({ behavior: 'smooth' });
}

// Batch Prediction Functions
function addBatchRow() {
    batchRowCount++;
    const tbody = document.getElementById('batch-tbody');
    const row = document.createElement('tr');
    row.id = `batch-row-${batchRowCount}`;
    
    row.innerHTML = `
        <td><input type="text" class="form-control product-id" required placeholder="PROD_0001"></td>
        <td><input type="date" class="form-control date" required></td>
        <td>
            <select class="form-control region" required>
                <option value="">Select Region</option>
                <option value="Region_1">Region 1</option>
                <option value="Region_2">Region 2</option>
                <option value="Region_3">Region 3</option>
                <option value="Region_4">Region 4</option>
                <option value="Region_5">Region 5</option>
            </select>
        </td>
        <td><input type="number" class="form-control price" step="0.01" placeholder="100.00"></td>
        <td>
            <select class="form-control category">
                <option value="">Select Category</option>
                <option value="Electronics">Electronics</option>
                <option value="Clothing">Clothing</option>
                <option value="Food">Food</option>
                <option value="Home">Home</option>
                <option value="Sports">Sports</option>
                <option value="Books">Books</option>
                <option value="Toys">Toys</option>
            </select>
        </td>
        <td>
            <button class="btn btn-sm btn-danger" onclick="removeBatchRow(${batchRowCount})">
                <i class="fas fa-trash"></i>
            </button>
        </td>
    `;
    
    tbody.appendChild(row);
    
    // Set default date
    const dateInput = row.querySelector('.date');
    dateInput.valueAsDate = new Date();
}

function removeBatchRow(rowId) {
    const row = document.getElementById(`batch-row-${rowId}`);
    if (row) {
        row.remove();
    }
}

function clearBatchForm() {
    document.getElementById('batch-tbody').innerHTML = '';
    batchRowCount = 0;
    addBatchRow();
}

function loadSampleData() {
    clearBatchForm();
    
    const sampleData = [
        { product_id: 'PROD_0001', date: '2024-01-15', region: 'Region_1', price: 100.0, category: 'Electronics' },
        { product_id: 'PROD_0002', date: '2024-01-15', region: 'Region_2', price: 50.0, category: 'Clothing' },
        { product_id: 'PROD_0003', date: '2024-01-16', region: 'Region_3', price: 25.0, category: 'Food' },
        { product_id: 'PROD_0004', date: '2024-01-16', region: 'Region_4', price: 75.0, category: 'Home' },
        { product_id: 'PROD_0005', date: '2024-01-17', region: 'Region_5', price: 30.0, category: 'Sports' }
    ];
    
    sampleData.forEach(data => {
        addBatchRow();
        const lastRow = document.getElementById(`batch-row-${batchRowCount}`);
        lastRow.querySelector('.product-id').value = data.product_id;
        lastRow.querySelector('.date').value = data.date;
        lastRow.querySelector('.region').value = data.region;
        lastRow.querySelector('.price').value = data.price;
        lastRow.querySelector('.category').value = data.category;
    });
}

async function predictBatch() {
    const rows = document.querySelectorAll('#batch-tbody tr');
    const predictions = [];
    
    // Validate and collect data
    for (const row of rows) {
        const product_id = row.querySelector('.product-id').value.trim();
        const date = row.querySelector('.date').value;
        const region = row.querySelector('.region').value;
        const price = parseFloat(row.querySelector('.price').value) || null;
        const category = row.querySelector('.category').value || null;
        
        if (!product_id || !date || !region) {
            showNotification('Please fill in all required fields (Product ID, Date, Region)', 'warning');
            row.classList.add('table-warning');
            return;
        }
        
        row.classList.remove('table-warning');
        predictions.push({
            product_id,
            date,
            region,
            price,
            category
        });
    }
    
    if (predictions.length === 0) {
        showNotification('No data to predict', 'warning');
        return;
    }
    
    showLoading('Making batch predictions...');
    
    try {
        const result = await apiCall('/predict-demand/batch', 'POST', { predictions });
        displayBatchResults(result);
        showNotification(`Batch prediction completed! ${result.total_predictions} predictions processed.`, 'success');
    } catch (error) {
        showNotification('Batch prediction failed: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

function displayBatchResults(result) {
    document.getElementById('total-predictions').textContent = result.total_predictions;
    document.getElementById('processing-time').textContent = result.processing_time_ms;
    
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    result.predictions.forEach(prediction => {
        const row = document.createElement('tr');
        const confidenceInterval = prediction.confidence_interval 
            ? `${prediction.confidence_interval.lower.toFixed(1)} - ${prediction.confidence_interval.upper.toFixed(1)}`
            : 'N/A';
        
        row.innerHTML = `
            <td>${prediction.product_id}</td>
            <td>${prediction.date}</td>
            <td>${prediction.region}</td>
            <td><strong>${prediction.predicted_demand}</strong></td>
            <td>${confidenceInterval}</td>
            <td>${prediction.model_version}</td>
        `;
        tbody.appendChild(row);
    });
    
    document.getElementById('batch-results').style.display = 'block';
    document.getElementById('batch-results').scrollIntoView({ behavior: 'smooth' });
}

// Drift Detection
async function triggerDriftDetection() {
    showLoading('Running drift detection...');
    
    try {
        const result = await apiCall('/drift-detect', 'POST');
        document.getElementById('drift-result').innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                ${result.message}
            </div>
        `;
        showNotification('Drift detection initiated successfully', 'success');
    } catch (error) {
        document.getElementById('drift-result').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Drift detection failed: ${error.message}
            </div>
        `;
        showNotification('Drift detection failed: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Utility Functions
function clearSingleForm() {
    document.getElementById('single-prediction-form').reset();
    document.getElementById('date').valueAsDate = new Date();
    document.getElementById('single-result').style.display = 'none';
}

function showLoading(message = 'Loading...') {
    console.log('Showing loading overlay:', message);
    const overlay = document.getElementById('loadingOverlay');
    const messageElement = document.getElementById('loading-message');
    
    if (overlay && messageElement) {
        messageElement.textContent = message;
        overlay.style.display = 'flex';
        console.log('Loading overlay shown');
    }
}

function hideLoading() {
    console.log('Hiding loading overlay...');
    const overlay = document.getElementById('loadingOverlay');
    
    if (overlay) {
        overlay.style.display = 'none';
        console.log('Loading overlay hidden');
    }
}

function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 350px;';
        document.body.appendChild(container);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to submit single prediction form
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const activeTab = document.querySelector('.tab-pane.active').id;
        if (activeTab === 'single') {
            document.getElementById('single-prediction-form').dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to close loading modal
    if (event.key === 'Escape') {
        hideLoading();
    }
});

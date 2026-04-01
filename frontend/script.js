// Global variables
const API_BASE_URL = 'http://localhost:8000';
let currentMode = 'single';
let selectedFiles = [];
let currentResults = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    setupEventListeners();
});

// Check API status
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');

        if (data.status === 'healthy' && data.model_loaded) {
            indicator.classList.add('online');
            statusText.textContent = 'System Online';
        } else {
            indicator.classList.add('offline');
            statusText.textContent = 'Model Not Loaded';
        }
    } catch (error) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        indicator.classList.add('offline');
        statusText.textContent = 'API Offline';
        console.error('API Status Check Failed:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    });
}

// Set mode (single/multi)
function setMode(mode) {
    currentMode = mode;

    const singleBtn = document.getElementById('singleModeBtn');
    const multiBtn = document.getElementById('multiModeBtn');
    const fileInput = document.getElementById('fileInput');
    const aggregationOptions = document.getElementById('aggregationOptions');

    if (mode === 'single') {
        singleBtn.classList.add('active');
        multiBtn.classList.remove('active');
        fileInput.removeAttribute('multiple');
        aggregationOptions.style.display = 'none';
    } else {
        multiBtn.classList.add('active');
        singleBtn.classList.remove('active');
        fileInput.setAttribute('multiple', 'multiple');
        aggregationOptions.style.display = 'block';
    }

    clearImages();
}

// Handle file selection
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    handleFiles(files);
}

// Handle files
function handleFiles(files) {
    if (currentMode === 'single' && files.length > 1) {
        alert('Single image mode: Please select only one image');
        return;
    }

    if (files.length > 10) {
        alert('Maximum 10 images allowed');
        return;
    }

    selectedFiles = files;
    displayPreviews();

    document.getElementById('actionButtons').style.display = 'flex';
}

// Display image previews
function displayPreviews() {
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';

    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `
                <img src="${e.target.result}" alt="Preview ${index + 1}">
                <button class="preview-remove" onclick="removeImage(${index})">×</button>
            `;
            container.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
}

// Remove image
function removeImage(index) {
    const filesArray = Array.from(selectedFiles);
    filesArray.splice(index, 1);
    selectedFiles = filesArray;

    if (selectedFiles.length === 0) {
        document.getElementById('actionButtons').style.display = 'none';
        document.getElementById('previewContainer').innerHTML = '';
    } else {
        displayPreviews();
    }
}

// Clear all images
function clearImages() {
    selectedFiles = [];
    document.getElementById('previewContainer').innerHTML = '';
    document.getElementById('actionButtons').style.display = 'none';
    document.getElementById('fileInput').value = '';
}

// Predict images
async function predictImages() {
    if (selectedFiles.length === 0) {
        alert('Please select at least one image');
        return;
    }

    showLoading(true);

    try {
        if (currentMode === 'single') {
            await predictSingle();
        } else {
            await predictMulti();
        }
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Prediction failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Predict single image
async function predictSingle() {
    const formData = new FormData();
    formData.append('file', selectedFiles[0]);

    const response = await fetch(`${API_BASE_URL}/predict_single`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }

    const result = await response.json();
    currentResults = result;
    displayResults(result);
}

// Predict multiple images
async function predictMulti() {
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    const aggregation = document.getElementById('aggregationMethod').value;
    formData.append('aggregation', aggregation);

    const response = await fetch(`${API_BASE_URL}/predict_multi`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }

    const result = await response.json();
    currentResults = result;
    displayResults(result);
}

// Display results
function displayResults(result) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

    // Final prediction
    document.getElementById('breedName').textContent = result.final_prediction;
    document.getElementById('animalType').textContent = result.animal_type.toUpperCase();

    // Confidence
    const confidence = Math.round(result.confidence * 100);
    document.getElementById('confidenceBar').style.width = confidence + '%';
    document.getElementById('confidenceText').textContent = confidence + '%';

    // Decision badge
    const badge = document.getElementById('decisionBadge');
    badge.textContent = result.decision;
    badge.className = 'decision-badge ' + result.decision;

    // Decision support
    document.getElementById('decisionMessage').textContent = result.decision_message;
    document.getElementById('recommendation').textContent = '💡 ' + result.recommendation;

    // Reasoning
    document.getElementById('reasoning').textContent = result.reasoning;

    // Top predictions
    displayTopPredictions(result.top_predictions);

    // Breed info
    if (result.breed_info && Object.keys(result.breed_info).length > 0) {
        displayBreedInfo(result.breed_info);
    }
}

// Display top predictions
function displayTopPredictions(predictions) {
    const container = document.getElementById('predictionsList');
    container.innerHTML = '';

    predictions.forEach(pred => {
        const confidence = Math.round(pred.confidence * 100);
        const div = document.createElement('div');
        div.className = 'prediction-item';
        div.innerHTML = `
            <div class="prediction-rank">#${pred.rank}</div>
            <div class="prediction-details">
                <div class="prediction-breed">${pred.breed}</div>
                <div class="prediction-bar">
                    <div class="prediction-bar-fill" style="width: ${confidence}%"></div>
                </div>
            </div>
            <div class="prediction-confidence">${confidence}%</div>
        `;
        container.appendChild(div);
    });
}

// Display breed information
function displayBreedInfo(info) {
    const container = document.getElementById('infoGrid');
    const section = document.getElementById('breedInfo');

    container.innerHTML = '';
    section.style.display = 'block';

    const fields = [
        { label: 'Description', value: info.description },
        { label: 'Origin', value: info.origin },
        { label: 'Milk Yield', value: info.milk_yield },
        { label: 'Key Features', value: info.key_features ? info.key_features.join(', ') : 'N/A' }
    ];

    fields.forEach(field => {
        if (field.value) {
            const div = document.createElement('div');
            div.className = 'info-item';
            div.innerHTML = `
                <div class="info-label">${field.label}</div>
                <div class="info-value">${field.value}</div>
            `;
            container.appendChild(div);
        }
    });
}

// Download results
function downloadResults() {
    if (!currentResults) {
        alert('No results to download');
        return;
    }

    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `breed_prediction_${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
}

// Reset app
function resetApp() {
    clearImages();
    document.getElementById('resultsSection').style.display = 'none';
    currentResults = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show/hide loading
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

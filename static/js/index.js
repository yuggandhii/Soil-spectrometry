// Main application state
const appState = {
    charts: {},
    cropRecommendations: {
        wheat: {
            seedling: {
                irrigation: "Maintain soil moisture at 60-70%",
                fertilizer: "Apply 20-30 kg N/ha",
                amendments: "Add organic matter if pH < 6.0"
            }
        }
    }
};

// Initialize the application
function initApp() {
    const app = document.getElementById('app');
    
    // Create main layout
    app.innerHTML = `
        <div class="container-fluid">
            <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">Soil Health Dashboard</a>
                </div>
            </nav>
            
            <div class="row mt-4">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Data Input</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label for="fileUpload" class="form-label">Upload Data File</label>
                                <input type="file" class="form-control" id="fileUpload" accept=".csv,.xlsx,.xls">
                            </div>
                            <button class="btn btn-primary" onclick="handleFileUpload()">Upload</button>
                            
                            <hr>
                            
                            <h6>Manual Data Entry</h6>
                            <div id="manualEntryForm"></div>
                            <button class="btn btn-success mt-3" onclick="handleManualEntry()">Submit</button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-8">
                    <div id="dashboardCards"></div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">Feature Importance Analysis</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4">
                                    <h6>Soil Quality Features</h6>
                                    <img src="/plot/feature_importance_soil_quality.png" 
                                         class="img-fluid" 
                                         alt="Soil Quality Feature Importance"
                                         onerror="this.onerror=null; this.src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';">
                                </div>
                                <div class="col-md-4">
                                    <h6>Crop Suitability Features</h6>
                                    <img src="/plot/feature_importance_crop_suitability.png" 
                                         class="img-fluid" 
                                         alt="Crop Suitability Feature Importance"
                                         onerror="this.onerror=null; this.src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';">
                                </div>
                                <div class="col-md-4">
                                    <h6>Yield Potential Features</h6>
                                    <img src="/plot/feature_importance_yield_potential.png" 
                                         class="img-fluid" 
                                         alt="Yield Potential Feature Importance"
                                         onerror="this.onerror=null; this.src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Initialize manual entry form
    generateManualEntryForm();
}

// Generate manual entry form fields
function generateManualEntryForm() {
    const form = document.getElementById('manualEntryForm');
    const fields = [
        { id: 'location_name', label: 'Location Name', type: 'text' },
        { id: 'date', label: 'Date', type: 'date' },
        { id: 'latitude', label: 'Latitude', type: 'number', step: '0.000001' },
        { id: 'longitude', label: 'Longitude', type: 'number', step: '0.000001' },
        { id: 'moisture', label: 'Soil Moisture (%)', type: 'number', step: '0.1' },
        { id: 'ph', label: 'pH Level', type: 'number', step: '0.1' },
        { id: 'temperature', label: 'Temperature (°C)', type: 'number', step: '0.1' },
        { id: 'salinity', label: 'Salinity (dS/m)', type: 'number', step: '0.01' }
    ];
    
    const nutrients = ['N', 'P', 'K', 'Ca', 'Mg'];
    
    fields.forEach(field => {
        const div = document.createElement('div');
        div.className = 'mb-3';
        div.innerHTML = `
            <label for="${field.id}" class="form-label">${field.label}</label>
            <input type="${field.type}" class="form-control" id="${field.id}" 
                   step="${field.step || '1'}" required>
        `;
        form.appendChild(div);
    });
    
    const nutrientsDiv = document.createElement('div');
    nutrientsDiv.className = 'mb-3';
    nutrientsDiv.innerHTML = `
        <label class="form-label">Nutrients (mg/kg)</label>
        ${nutrients.map(nutrient => `
            <div class="input-group mb-2">
                <span class="input-group-text">${nutrient}</span>
                <input type="number" class="form-control" id="nutrient_${nutrient}" 
                       step="0.1" required>
            </div>
        `).join('')}
    `;
    form.appendChild(nutrientsDiv);
}

// Handle file upload
async function handleFileUpload() {
    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file to upload');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateDashboard(data.results);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while uploading the file');
    }
}

// Handle manual data entry
async function handleManualEntry() {
    const formData = {
        location_name: document.getElementById('location_name').value,
        date: document.getElementById('date').value,
        latitude: parseFloat(document.getElementById('latitude').value),
        longitude: parseFloat(document.getElementById('longitude').value),
        moisture: parseFloat(document.getElementById('moisture').value),
        ph: parseFloat(document.getElementById('ph').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        salinity: parseFloat(document.getElementById('salinity').value),
        nutrients: {
            N: parseFloat(document.getElementById('nutrient_N').value),
            P: parseFloat(document.getElementById('nutrient_P').value),
            K: parseFloat(document.getElementById('nutrient_K').value),
            Ca: parseFloat(document.getElementById('nutrient_Ca').value),
            Mg: parseFloat(document.getElementById('nutrient_Mg').value)
        }
    };
    
    try {
        const response = await fetch('/add_data_point', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateDashboard(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while submitting the data');
    }
}

// Update dashboard with new data
function updateDashboard(data) {
    const dashboardCards = document.getElementById('dashboardCards');
    
    // Create soil health indicators card
    const soilHealthCard = createSoilHealthCard(data.soil_health);
    dashboardCards.appendChild(soilHealthCard);
    
    // Create predictions card
    const predictionsCard = createPredictionsCard(data.predictions);
    dashboardCards.appendChild(predictionsCard);
}

// Create soil health indicators card
function createSoilHealthCard(soilHealth) {
    const card = document.createElement('div');
    card.className = 'card mb-3';
    card.innerHTML = `
        <div class="card-header">
            <h5 class="card-title mb-0">Soil Health Indicators</h5>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label">Soil Moisture</label>
                        <div class="progress">
                            <div class="progress-bar ${getMoistureClass(soilHealth.moisture)}" 
                                 role="progressbar" 
                                 style="width: ${soilHealth.moisture}%">
                                ${soilHealth.moisture.toFixed(1)}%
                            </div>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">pH Level</label>
                        <div class="progress">
                            <div class="progress-bar ${getPHClass(soilHealth.ph)}" 
                                 role="progressbar" 
                                 style="width: ${(soilHealth.ph / 14) * 100}%">
                                ${soilHealth.ph.toFixed(1)}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label class="form-label">Temperature</label>
                        <div class="progress">
                            <div class="progress-bar ${getTemperatureClass(soilHealth.temperature)}" 
                                 role="progressbar" 
                                 style="width: ${(soilHealth.temperature / 40) * 100}%">
                                ${soilHealth.temperature.toFixed(1)}°C
                            </div>
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Salinity</label>
                        <div class="progress">
                            <div class="progress-bar ${getSalinityClass(soilHealth.salinity)}" 
                                 role="progressbar" 
                                 style="width: ${(soilHealth.salinity / 4) * 100}%">
                                ${soilHealth.salinity.toFixed(2)} dS/m
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-3">
                <h6>Nutrient Levels</h6>
                <div class="row">
                    ${Object.entries(soilHealth.nutrients).map(([nutrient, value]) => `
                        <div class="col-md-4 mb-2">
                            <label class="form-label">${nutrient}</label>
                            <div class="progress">
                                <div class="progress-bar ${getNutrientClass(nutrient, value)}" 
                                     role="progressbar" 
                                     style="width: ${(value / 100) * 100}%">
                                    ${value.toFixed(1)} mg/kg
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    return card;
}

// Create predictions card
function createPredictionsCard(predictions) {
    const card = document.createElement('div');
    card.className = 'card mb-3';
    card.innerHTML = `
        <div class="card-header">
            <h5 class="card-title mb-0">Predictions</h5>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-4">
                    <div class="mb-3">
                        <label class="form-label">Soil Quality</label>
                        <div class="progress">
                            <div class="progress-bar ${getPredictionClass(predictions.soil_quality)}" 
                                 role="progressbar" 
                                 style="width: ${predictions.soil_quality}%">
                                ${predictions.soil_quality.toFixed(1)}%
                            </div>
                        </div>
                        <small class="text-muted">Confidence: ${predictions.confidence.soil_quality.toFixed(1)}%</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="mb-3">
                        <label class="form-label">Crop Suitability</label>
                        <div class="progress">
                            <div class="progress-bar ${getPredictionClass(predictions.crop_suitability)}" 
                                 role="progressbar" 
                                 style="width: ${predictions.crop_suitability}%">
                                ${predictions.crop_suitability.toFixed(1)}%
                            </div>
                        </div>
                        <small class="text-muted">Confidence: ${predictions.confidence.crop_suitability.toFixed(1)}%</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="mb-3">
                        <label class="form-label">Yield Potential</label>
                        <div class="progress">
                            <div class="progress-bar ${getPredictionClass(predictions.yield_potential)}" 
                                 role="progressbar" 
                                 style="width: ${predictions.yield_potential}%">
                                ${predictions.yield_potential.toFixed(1)}%
                            </div>
                        </div>
                        <small class="text-muted">Confidence: ${predictions.confidence.yield_potential.toFixed(1)}%</small>
                    </div>
                </div>
            </div>
        </div>
    `;
    return card;
}

// Helper functions for styling
function getMoistureClass(value) {
    if (value < 30) return 'bg-danger';
    if (value < 60) return 'bg-warning';
    return 'bg-success';
}

function getPHClass(value) {
    if (value < 6.0) return 'bg-danger';
    if (value < 7.0) return 'bg-warning';
    return 'bg-success';
}

function getTemperatureClass(value) {
    if (value < 10) return 'bg-danger';
    if (value < 25) return 'bg-warning';
    return 'bg-success';
}

function getSalinityClass(value) {
    if (value > 2) return 'bg-danger';
    if (value > 1) return 'bg-warning';
    return 'bg-success';
}

function getNutrientClass(nutrient, value) {
    const thresholds = {
        N: { warning: 20, danger: 10 },
        P: { warning: 15, danger: 8 },
        K: { warning: 150, danger: 100 },
        Ca: { warning: 1000, danger: 500 },
        Mg: { warning: 150, danger: 100 }
    };
    
    if (value < thresholds[nutrient].danger) return 'bg-danger';
    if (value < thresholds[nutrient].warning) return 'bg-warning';
    return 'bg-success';
}

function getPredictionClass(value) {
    if (value < 50) return 'bg-danger';
    if (value < 75) return 'bg-warning';
    return 'bg-success';
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', initApp); 
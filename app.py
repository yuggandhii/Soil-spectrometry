from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
from flask_debugtoolbar import DebugToolbarExtension
from soil_spectroscopy_analysis import (
    load_data, analyze_sensor_correlations, build_sensor_prediction_models,
    create_spectral_indices, analyze_wavelength_regions, improve_temperature_prediction,
    load_models, predict_soil_health, train_new_models, analyze_spectra,
    plot_feature_importance
)
import json
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask_sqlalchemy import SQLAlchemy
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
toolbar = DebugToolbarExtension(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///soil_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class SoilData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(100))
    moisture = db.Column(db.Float)
    ph = db.Column(db.Float)
    temperature = db.Column(db.Float)
    salinity = db.Column(db.Float)
    n = db.Column(db.Float)
    p = db.Column(db.Float)
    k = db.Column(db.Float)
    ca = db.Column(db.Float)
    mg = db.Column(db.Float)
    soil_quality = db.Column(db.Float)
    crop_suitability = db.Column(db.Float)
    yield_potential = db.Column(db.Float)

# Create database tables
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

class TemplateReloader(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app

    def on_modified(self, event):
        if event.src_path.endswith('.html'):
            logger.info(f"Template file changed: {event.src_path}")
            # Reload the template
            self.app.jinja_env.cache.clear()
            logger.info("Template cache cleared")

def setup_template_reloader(app):
    observer = Observer()
    observer.schedule(TemplateReloader(app), path='templates', recursive=False)
    observer.start()
    logger.info("Template reloader started")
    return observer

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_time_series_data(df, indicator):
    """Generate time series data for a given indicator"""
    try:
        # Create time series data
        time_series = df[indicator].tolist()
        timestamps = pd.date_range(start='2024-01-01', periods=len(time_series), freq='D').strftime('%Y-%m-%d').tolist()
        
        return {
            'labels': timestamps,
            'values': time_series
        }
    except Exception as e:
        logger.error(f"Error generating time series data: {str(e)}")
        return {'labels': [], 'values': []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_data_point', methods=['POST'])
def add_data_point():
    try:
        data = request.get_json()
        logger.debug(f"Received data point: {data}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['location_name', 'date', 'latitude', 'longitude', 'moisture', 
                         'ph', 'temperature', 'salinity', 'nutrients']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate numeric fields
        numeric_fields = ['moisture', 'ph', 'temperature', 'salinity', 'N', 'P', 'K', 'Ca', 'Mg']
        for field in numeric_fields:
            try:
                value = float(data[field])
                if field == 'moisture' and (value < 0 or value > 100):
                    return jsonify({'error': f'Moisture must be between 0 and 100'}), 400
                elif field == 'ph' and (value < 0 or value > 14):
                    return jsonify({'error': f'pH must be between 0 and 14'}), 400
                elif field == 'temperature' and (value < -50 or value > 100):
                    return jsonify({'error': f'Temperature must be between -50°C and 100°C'}), 400
                elif field == 'salinity' and (value < 0 or value > 100):
                    return jsonify({'error': f'Salinity must be between 0 and 100'}), 400
                elif field in ['N', 'P', 'K', 'Ca', 'Mg'] and value < 0:
                    return jsonify({'error': f'{field} must be non-negative'}), 400
            except ValueError:
                return jsonify({'error': f'Invalid value for {field}'}), 400
        
        # Create feature array for ML prediction
        features = np.array([[
            float(data['moisture']),
            float(data['ph']),
            float(data['temperature']),
            float(data['salinity']),
            float(data['nutrients']['N']),
            float(data['nutrients']['P']),
            float(data['nutrients']['K']),
            float(data['nutrients']['Ca']),
            float(data['nutrients']['Mg'])
        ]])
        
        # Get ML predictions
        models = train_new_models()
        predictions = predict_soil_health(features, models)
        
        # Create new soil data entry
        soil_data = SoilData(
            location=data['location_name'],
            moisture=float(data['moisture']),
            ph=float(data['ph']),
            temperature=float(data['temperature']),
            salinity=float(data['salinity']),
            n=float(data['nutrients']['N']),
            p=float(data['nutrients']['P']),
            k=float(data['nutrients']['K']),
            ca=float(data['nutrients']['Ca']),
            mg=float(data['nutrients']['Mg']),
            soil_quality=float(predictions['soil_quality']),
            crop_suitability=float(predictions['crop_suitability']),
            yield_potential=float(predictions['yield_potential'])
        )
        
        # Save to database
        db.session.add(soil_data)
        db.session.commit()
        
        # Prepare response data
        response_data = {
            'soil_health': {
                'moisture': float(data['moisture']),
                'ph': float(data['ph']),
                'temperature': float(data['temperature']),
                'salinity': float(data['salinity']),
                'nutrients': {
                    'total': sum(float(data[n]) for n in ['N', 'P', 'K', 'Ca', 'Mg']),
                    'N': float(data['nutrients']['N']),
                    'P': float(data['nutrients']['P']),
                    'K': float(data['nutrients']['K']),
                    'Ca': float(data['nutrients']['Ca']),
                    'Mg': float(data['nutrients']['Mg'])
                }
            },
            'time_series': {
                'moisture': generate_time_series_data(pd.DataFrame([data]), 'moisture'),
                'ph': generate_time_series_data(pd.DataFrame([data]), 'ph'),
                'temperature': generate_time_series_data(pd.DataFrame([data]), 'temperature'),
                'salinity': generate_time_series_data(pd.DataFrame([data]), 'salinity')
            },
            'predictions': predictions
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error adding data point: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.debug("Received upload request")
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.debug(f"Saving file to: {filepath}")
            file.save(filepath)
            
            try:
                # Debug point 1: Check file loading
                logger.debug(f"Loading file: {filepath}")
                df = load_data(filepath)
                if df is None:
                    logger.error("Failed to load data")
                    return jsonify({'error': 'Error loading data'}), 400
                
                # Debug point 2: Check spectral columns
                spectral_cols = [str(wl) for wl in range(410, 941, 5) if str(wl) in df.columns]
                logger.debug(f"Found {len(spectral_cols)} spectral columns")
                
                # Perform analysis
                results = {
                    'wavelength_regions': {},
                    'sensor_correlations': {},
                    'model_results': {},
                    'spectral_indices': {},
                    'soil_health': {},
                    'time_series': {},
                    'plots': {}
                }
                
                # Calculate soil health indicators
                results['soil_health'] = {
                    'moisture': df['moisture'].mean(),
                    'ph': df['ph'].mean(),
                    'temperature': df['temperature'].mean(),
                    'salinity': df['salinity'].mean(),
                    'nutrients': {
                        'total': df[['N', 'P', 'K', 'Ca', 'Mg']].sum(axis=1).mean(),
                        'N': df['N'].mean(),
                        'P': df['P'].mean(),
                        'K': df['K'].mean(),
                        'Ca': df['Ca'].mean(),
                        'Mg': df['Mg'].mean()
                    }
                }
                
                # Generate time series data
                for indicator in ['moisture', 'ph', 'temperature', 'salinity']:
                    results['time_series'][indicator] = generate_time_series_data(df, indicator)
                
                # Debug point 3: Analyze wavelength regions
                logger.debug("Analyzing wavelength regions...")
                analyze_wavelength_regions(df, spectral_cols)
                
                # Debug point 4: Create spectral indices
                logger.debug("Creating spectral indices...")
                indices_df = create_spectral_indices(df, spectral_cols)
                results['spectral_indices'] = indices_df.head().to_dict()
                
                # Debug point 5: Analyze sensor correlations
                logger.debug("Analyzing sensor correlations...")
                analyze_sensor_correlations(df)
                
                # Debug point 6: Build prediction models
                logger.debug("Building prediction models...")
                build_sensor_prediction_models(df)
                
                # Debug point 7: Improve temperature prediction
                logger.debug("Improving temperature prediction...")
                improve_temperature_prediction(df)
                
                # Generate ML model predictions
                try:
                    # Create feature array for prediction
                    features = np.array([[
                        df['moisture'].mean(),
                        df['ph'].mean(),
                        df['temperature'].mean(),
                        df['salinity'].mean(),
                        df['N'].mean(),
                        df['P'].mean(),
                        df['K'].mean(),
                        df['Ca'].mean(),
                        df['Mg'].mean()
                    ]])
                    
                    # Load and run ML models
                    models = train_new_models()
                    predictions = predict_soil_health(features, models)
                    
                    # Add predictions to results
                    results['predictions'] = predictions
                    
                except Exception as e:
                    logger.error(f"ML model prediction error: {str(e)}")
                    results['predictions'] = None
                
                # Generate plots
                logger.debug("Generating plots...")
                
                # Generate average spectrum plot
                plt.figure(figsize=(12, 6))
                plt.plot(df[spectral_cols].mean())
                plt.title('Average Spectral Response')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Reflectance')
                plt.tight_layout()
                plt.savefig('average_spectrum.png')
                plt.close()
                
                # Generate nutrient distribution plot
                plt.figure(figsize=(10, 6))
                nutrients = ['N', 'P', 'K', 'Ca', 'Mg']
                nutrient_values = [df[nutrient].mean() for nutrient in nutrients]
                plt.bar(nutrients, nutrient_values)
                plt.title('Average Nutrient Levels')
                plt.xlabel('Nutrient')
                plt.ylabel('Concentration (mg/kg)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('nutrient_distribution.png')
                plt.close()
                
                # Generate soil health indicators plot
                plt.figure(figsize=(10, 6))
                indicators = ['moisture', 'ph', 'temperature', 'salinity']
                indicator_values = [df[indicator].mean() for indicator in indicators]
                plt.bar(indicators, indicator_values)
                plt.title('Average Soil Health Indicators')
                plt.xlabel('Indicator')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('soil_health_indicators.png')
                plt.close()
                
                # Get generated plots
                plots = {
                    'average_spectrum': 'average_spectrum.png',
                    'sensor_correlations': 'sensor_correlations.png',
                    'nutrient_distribution': 'nutrient_distribution.png',
                    'soil_health_indicators': 'soil_health_indicators.png'
                }
                
                # Add region-specific plots
                for region in ['visible', 'nir', 'temperature_sensitive', 'moisture_sensitive']:
                    plot_name = f'spectrum_{region}.png'
                    if os.path.exists(plot_name):
                        plots[region] = plot_name
                
                results['plots'].update(plots)
                
                # Generate feature importance plots
                logger.debug("Generating feature importance plots...")
                feature_names = ['Moisture', 'pH', 'Temperature', 'Salinity', 'N', 'P', 'K', 'Ca', 'Mg']
                plot_feature_importance(
                    models['soil_quality'],
                    feature_names,
                    'Feature Importance for Soil Quality Prediction',
                    'static/plots/feature_importance_soil_quality.png'
                )
                plot_feature_importance(
                    models['crop_suitability'],
                    feature_names,
                    'Feature Importance for Crop Suitability Prediction',
                    'static/plots/feature_importance_crop_suitability.png'
                )
                plot_feature_importance(
                    models['yield_potential'],
                    feature_names,
                    'Feature Importance for Yield Potential Prediction',
                    'static/plots/feature_importance_yield_potential.png'
                )
                
                logger.debug("Analysis completed successfully")
                return jsonify({
                    'success': True,
                    'message': 'Analysis completed successfully',
                    'plots': results['plots'],
                    'results': results
                })
                
            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Cleaned up file: {filepath}")
        
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/plot/<filename>')
def serve_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == '__main__':
    # Set up template reloader
    observer = setup_template_reloader(app)
    
    try:
        # Enable debug mode with auto-reload
        app.run(debug=True, use_reloader=True, host='0.0.0.0', port=5000)
    finally:
        observer.stop()
        observer.join() 
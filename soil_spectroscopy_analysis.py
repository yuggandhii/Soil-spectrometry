import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import re
import joblib
import os

def sanitize_filename(filename):
    """
    Sanitize filename by removing special characters and spaces
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.strip('-_')

def load_data(filepath):
    """Load and preprocess the soil spectroscopy data"""
    try:
        # Try to read the file
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        
        # Define column mappings for different possible formats
        column_mappings = {
            'moisture': ['moisture', 'Soil Moisture (%)', 'soil_moisture', 'moisture_content', 'Soil_Moisture'],
            'ph': ['ph', 'pH Level', 'ph_level', 'soil_ph', 'pH'],
            'temperature': ['temperature', 'Temperature (°C)', 'temp', 'soil_temperature', 'Temperature'],
            'salinity': ['salinity', 'Salinity (dS/m)', 'soil_salinity', 'Salinity'],
            'N': ['N', 'Nitrogen (mg/kg)', 'nitrogen', 'N_content', 'Nitrogen'],
            'P': ['P', 'Phosphorus (mg/kg)', 'phosphorus', 'P_content', 'Phosphorus'],
            'K': ['K', 'Potassium (mg/kg)', 'potassium', 'K_content', 'Potassium'],
            'Ca': ['Ca', 'Calcium (mg/kg)', 'calcium', 'Ca_content', 'Calcium'],
            'Mg': ['Mg', 'Magnesium (mg/kg)', 'magnesium', 'Mg_content', 'Magnesium']
        }
        
        # Create a mapping of existing columns to standard names
        rename_dict = {}
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                col_lower = str(col).lower()
                for possible_name in possible_names:
                    if possible_name.lower() in col_lower:
                        rename_dict[col] = standard_name
                        break
                if col in rename_dict:
                    break
        
        # Rename columns if matches found
        if rename_dict:
            print("Renaming columns:", rename_dict)
            df = df.rename(columns=rename_dict)
        
        # Check for required columns
        required_columns = ['moisture', 'ph', 'temperature', 'salinity', 'N', 'P', 'K', 'Ca', 'Mg']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert numeric columns
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with column means
        df = df.fillna(df.mean())
        
        # Validate data ranges
        if not (0 <= df['moisture'].mean() <= 100):
            raise ValueError("Moisture values must be between 0 and 100")
        if not (0 <= df['ph'].mean() <= 14):
            raise ValueError("pH values must be between 0 and 14")
        if not (-50 <= df['temperature'].mean() <= 100):
            raise ValueError("Temperature values must be between -50°C and 100°C")
        if not (0 <= df['salinity'].mean() <= 100):
            raise ValueError("Salinity values must be between 0 and 100")
        
        # Validate nutrient values
        for nutrient in ['N', 'P', 'K', 'Ca', 'Mg']:
            if (df[nutrient] < 0).any():
                raise ValueError(f"{nutrient} values cannot be negative")
        
        print("Processed data shape:", df.shape)
        print("Processed columns:", df.columns.tolist())
        print("Sample data:\n", df[required_columns].head())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(f"File path: {filepath}")
        if 'df' in locals():
            print("Available columns:", df.columns.tolist())
        return None

def explore_data(df):
    """
    Explore the data structure and content
    """
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nUnique values in each column:")
    for column in df.columns:
        print(f"\n{column}:")
        print(df[column].nunique(), "unique values")
        if df[column].nunique() < 10:  # Only show unique values for columns with few unique values
            print(df[column].unique())

def preprocess_data(df):
    """
    Preprocess the spectroscopic data
    """
    # Separate spectral features and target variables
    spectral_cols = [str(wl) for wl in range(410, 941, 5) if str(wl) in df.columns]
    target_cols = ['Ph', 'Nitro (mg/10 g)', 'Posh Nitro (mg/', 'Pota Nitro (mg/1']
    
    X = df[spectral_cols]
    y = df[target_cols]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, spectral_cols, target_cols

def analyze_spectra(df):
    """
    Perform spectral analysis
    """
    # Get spectral columns
    spectral_cols = [str(wl) for wl in range(410, 941, 5) if str(wl) in df.columns]
    
    # Create wavelength array
    wavelengths = np.array([int(wl) for wl in spectral_cols])
    
    # Plot average spectrum
    plt.figure(figsize=(12, 6))
    avg_spectrum = df[spectral_cols].mean()
    plt.plot(wavelengths, avg_spectrum, 'b-', label='Average Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Average Soil Spectral Reflectance')
    plt.grid(True)
    plt.legend()
    plt.savefig('average_spectrum.png')
    plt.close()
    
    # Calculate and plot spectral indices
    # NDVI-like index (using NIR and Red bands)
    nir_band = '860'
    red_band = '680'
    if nir_band in spectral_cols and red_band in spectral_cols:
        ndvi = (df[nir_band] - df[red_band]) / (df[nir_band] + df[red_band])
        plt.figure(figsize=(10, 6))
        plt.hist(ndvi, bins=50)
        plt.xlabel('Spectral Index Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Spectral Index')
        plt.savefig('spectral_index_distribution.png')
        plt.close()

def analyze_sensor_correlations(df):
    """Analyze correlations between different sensors"""
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()
        
        # Create correlation plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Sensor Correlations')
        plt.tight_layout()
        plt.savefig('sensor_correlations.png')
        plt.close()
        
        return correlations
    except Exception as e:
        print(f"Error analyzing correlations: {str(e)}")
        return None

def build_sensor_prediction_models(df):
    """Build prediction models for different soil parameters"""
    try:
        # Define target variables
        targets = ['moisture', 'ph', 'temperature', 'salinity']
        
        # Select feature columns (excluding targets)
        feature_cols = [col for col in df.columns if col not in targets]
        
        # Train models for each target
        models = {}
        for target in targets:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(df[feature_cols], df[target])
            models[target] = model
        
        return models
    except Exception as e:
        print(f"Error building prediction models: {str(e)}")
        return None

def create_spectral_indices(df, spectral_cols):
    """Create spectral indices from the data"""
    try:
        indices_df = pd.DataFrame()
        
        # Calculate NDVI (Normalized Difference Vegetation Index)
        if '800' in spectral_cols and '680' in spectral_cols:
            indices_df['NDVI'] = (df['800'] - df['680']) / (df['800'] + df['680'])
        
        # Calculate other indices as needed
        
        return indices_df
    except Exception as e:
        print(f"Error creating spectral indices: {str(e)}")
        return pd.DataFrame()

def analyze_wavelength_regions(df, spectral_cols):
    """Analyze different wavelength regions"""
    try:
        # Define wavelength regions
        regions = {
            'visible': [400, 700],
            'nir': [700, 1000],
            'temperature_sensitive': [800, 900],
            'moisture_sensitive': [1400, 1500]
        }
        
        # Analyze each region
        for region, (start, end) in regions.items():
            region_cols = [col for col in spectral_cols if start <= float(col) <= end]
            if region_cols:
                # Calculate statistics for the region
                stats = df[region_cols].describe()
                
                # Create visualization
                plt.figure(figsize=(10, 6))
                plt.plot([float(col) for col in region_cols], stats.loc['mean'])
                plt.title(f'Spectral Response in {region} Region')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Reflectance')
                plt.tight_layout()
                plt.savefig(f'spectrum_{region}.png')
                plt.close()
        
        return True
    except Exception as e:
        print(f"Error analyzing wavelength regions: {str(e)}")
        return False

def improve_temperature_prediction(df):
    """Improve temperature prediction using additional features"""
    try:
        # Add derived features
        df['temp_moisture_interaction'] = df['temperature'] * df['moisture']
        df['temp_salinity_interaction'] = df['temperature'] * df['salinity']
        
        # Create temperature prediction model
        feature_cols = ['moisture', 'salinity', 'temp_moisture_interaction', 'temp_salinity_interaction']
        target_col = 'temperature'
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df[feature_cols], df[target_col])
        
        return model
    except Exception as e:
        print(f"Error improving temperature prediction: {str(e)}")
        return None

def load_models():
    """Load the trained ML models"""
    try:
        # Check if models exist
        if not os.path.exists('models'):
            os.makedirs('models')
            return train_new_models()
        
        models = {}
        model_files = {
            'soil_quality': 'models/soil_quality_model.joblib',
            'crop_suitability': 'models/crop_suitability_model.joblib',
            'yield_potential': 'models/yield_potential_model.joblib'
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
            else:
                print(f"Model file not found: {filepath}")
                return train_new_models()
        
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return train_new_models()

def plot_feature_importance(model, feature_names, title, filename):
    """Generate and save feature importance plot"""
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(title)
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(filename)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating feature importance plot: {str(e)}")
        return False

def train_new_models():
    """Train new ML models with default parameters"""
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Initialize models with better parameters
        models = {
            'soil_quality': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'crop_suitability': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'yield_potential': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        # Create sample training data
        n_samples = 1000
        X = np.random.rand(n_samples, 9)  # 9 features
        X[:, 0] = X[:, 0] * 100  # moisture (0-100)
        X[:, 1] = X[:, 1] * 14   # pH (0-14)
        X[:, 2] = X[:, 2] * 150 - 50  # temperature (-50 to 100)
        X[:, 3] = X[:, 3] * 100  # salinity (0-100)
        X[:, 4:] = X[:, 4:] * 10  # nutrients (0-10)
        
        # Define feature names
        feature_names = ['Moisture', 'pH', 'Temperature', 'Salinity', 'N', 'P', 'K', 'Ca', 'Mg']
        
        # Generate target values based on soil health rules
        soil_quality = (
            0.3 * (X[:, 0] / 100) +  # moisture contribution
            0.2 * (1 - abs(X[:, 1] - 7) / 7) +  # pH contribution (optimal around 7)
            0.2 * (1 - abs(X[:, 2] - 25) / 75) +  # temperature contribution (optimal around 25°C)
            0.3 * (1 - X[:, 3] / 100)  # salinity contribution
        ) * 100
        
        crop_suitability = (
            0.4 * (X[:, 0] / 100) +  # moisture contribution
            0.3 * (1 - abs(X[:, 1] - 7) / 7) +  # pH contribution
            0.3 * (1 - X[:, 3] / 100)  # salinity contribution
        ) * 100
        
        yield_potential = (
            0.3 * (X[:, 0] / 100) +  # moisture contribution
            0.2 * (1 - abs(X[:, 1] - 7) / 7) +  # pH contribution
            0.2 * (1 - abs(X[:, 2] - 25) / 75) +  # temperature contribution
            0.3 * (np.sum(X[:, 4:], axis=1) / 50)  # nutrient contribution
        ) * 100
        
        # Train models
        models['soil_quality'].fit(X, soil_quality)
        models['crop_suitability'].fit(X, crop_suitability)
        models['yield_potential'].fit(X, yield_potential)
        
        # Generate feature importance plots
        plot_feature_importance(
            models['soil_quality'],
            feature_names,
            'Feature Importance for Soil Quality Prediction',
            'feature_importance_soil_quality.png'
        )
        plot_feature_importance(
            models['crop_suitability'],
            feature_names,
            'Feature Importance for Crop Suitability Prediction',
            'feature_importance_crop_suitability.png'
        )
        plot_feature_importance(
            models['yield_potential'],
            feature_names,
            'Feature Importance for Yield Potential Prediction',
            'feature_importance_yield_potential.png'
        )
        
        # Save models
        for name, model in models.items():
            joblib.dump(model, f'models/{name}_model.joblib')
        
        return models
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return None

def predict_soil_health(features, models):
    """Generate predictions using the ML models"""
    try:
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Generate predictions
        predictions = {
            'soil_quality': float(models['soil_quality'].predict(scaled_features)[0]),
            'crop_suitability': float(models['crop_suitability'].predict(scaled_features)[0]),
            'yield_potential': float(models['yield_potential'].predict(scaled_features)[0])
        }
        
        # Ensure predictions are within 0-100 range
        for key in predictions:
            predictions[key] = max(0, min(100, predictions[key]))
        
        # Add confidence scores based on feature values
        confidence = {
            'soil_quality': calculate_confidence(features[0], predictions['soil_quality']),
            'crop_suitability': calculate_confidence(features[0], predictions['crop_suitability']),
            'yield_potential': calculate_confidence(features[0], predictions['yield_potential'])
        }
        
        # Add confidence scores to predictions
        predictions['confidence'] = confidence
        
        return predictions
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return {
            'soil_quality': 50.0,
            'crop_suitability': 50.0,
            'yield_potential': 50.0,
            'confidence': {
                'soil_quality': 0.5,
                'crop_suitability': 0.5,
                'yield_potential': 0.5
            }
        }

def calculate_confidence(features, prediction):
    """Calculate confidence score based on feature values and prediction"""
    try:
        # Normalize features to 0-1 range
        features_norm = features.copy()
        features_norm[0] = features_norm[0] / 100  # moisture
        features_norm[1] = features_norm[1] / 14   # pH
        features_norm[2] = (features_norm[2] + 50) / 150  # temperature
        features_norm[3] = features_norm[3] / 100  # salinity
        features_norm[4:] = features_norm[4:] / 10  # nutrients
        
        # Calculate confidence based on feature values
        confidence = np.mean([
            1 - abs(features_norm[0] - 0.5),  # moisture confidence
            1 - abs(features_norm[1] - 0.5),  # pH confidence
            1 - abs(features_norm[2] - 0.5),  # temperature confidence
            1 - abs(features_norm[3] - 0.5),  # salinity confidence
            np.mean(1 - abs(features_norm[4:] - 0.5))  # nutrient confidence
        ])
        
        # Adjust confidence based on prediction value
        prediction_confidence = 1 - abs(prediction - 50) / 50
        
        # Combine confidences
        final_confidence = (confidence + prediction_confidence) / 2
        
        return float(final_confidence)
    except Exception as e:
        print(f"Error calculating confidence: {str(e)}")
        return 0.5

def main():
    # Load the data
    file_path = "soildataset.xlsx"
    df = load_data(file_path)
    
    if df is not None:
        # Explore the data
        explore_data(df)
        
        # Get spectral columns
        spectral_cols = [str(wl) for wl in range(410, 941, 5) if str(wl) in df.columns]
        
        # Analyze wavelength regions
        print("\nAnalyzing wavelength regions...")
        analyze_wavelength_regions(df, spectral_cols)
        
        # Create and analyze spectral indices
        print("\nCreating spectral indices...")
        indices_df = create_spectral_indices(df, spectral_cols)
        print("\nSpectral indices created:")
        print(indices_df.head())
        
        # Build combined sensor model
        print("\nBuilding combined sensor model...")
        build_sensor_prediction_models(df)
        
        # Improve temperature prediction
        print("\nImproving temperature prediction...")
        improve_temperature_prediction(df)
        
        # Original analysis
        print("\nPerforming original analysis...")
        analyze_sensor_correlations(df)
        
        # Preprocess the data for soil properties
        X_scaled, y, spectral_cols, target_cols = preprocess_data(df)
        analyze_spectra(df)
        
        # Load models
        models = load_models()
        
        # Predict soil health
        print("\nPredicting soil health...")
        predictions = predict_soil_health(X_scaled, models)
        print("\nSoil Health Predictions:")
        for target, prediction in predictions.items():
            print(f"{target}: {prediction:.2f}")

if __name__ == "__main__":
    main() 
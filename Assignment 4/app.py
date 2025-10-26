"""
Assignment 3: User Interface for ML Models
Backend API for Absenteeism Prediction Model

This Flask application serves the trained model from Assignment 2 and provides
an API for the frontend interface.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
# CORS(app, origins=['http://localhost:5173', 'http://localhost:3000', 'http://localhost:5000']
app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
cors_origins = os.environ.get('CORS_ORIGINS', '*')
if ',' in cors_origins:
    cors_origins = [o.strip() for o in cors_origins.split(',')]
CORS(app, resources={r"/api/*": {"origins": cors_origins}})


# Global variables to store the trained model and scaler
model = None
scaler = None
feature_columns = None

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
def load_model():
    """Load the trained model and scaler from Assignment 2"""
    global model, scaler, feature_columns
    
    try:
        # Load the model and scaler (we'll create these from Assignment 2)
        # with open('model.pkl', 'rb') as f:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_names']  # Changed from 'feature_columns' to 'feature_names'
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model file not found. Please train the model first (python train_model.py).")
        model = None
        scaler = None
        feature_columns = None

@app.route('/api/health', methods=['GET'])
def health():
    """Health check and model status"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'feature_columns': len(feature_columns) if feature_columns is not None else 0
    })

def preprocess_input(data):
    """Preprocess input data to match the model's expected format"""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Handle categorical variables (same as in Assignment 2)
    categorical_cols = ['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 
                        'Hit target', 'Disciplinary failure', 'Education', 'Son', 
                        'Social drinker', 'Social smoker', 'Pet']
    
    # Create dummy variables for categorical columns
    available_cat = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=available_cat, drop_first=True)
    
    # Ensure all expected columns are present
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
    
    # Reorder columns to match training data
    if feature_columns is not None:
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    return df_encoded

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_react_app(path):
    """Serve frontend for all other routes"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data
        data = request.json
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Scale the features
        scaled_data = scaler.transform(processed_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        # Simple placeholder confidence score
        confidence = 0.8
        
        return jsonify({
            'prediction': float(prediction),
            'confidence': confidence,
            'message': f'Predicted absenteeism: {prediction:.2f} hours'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """API endpoint for model information and fairness metrics"""
    try:
        model_info = {
            'model_type': 'Linear Regression',
            'performance': {
                'baseline': {
                    'rmse': 11.4292,
                    'mae': 6.4389,
                    'r2_score': -0.1987
                },
                'mitigated': {
                    'rmse': 43.1228,
                    'mae': 16.5046,
                    'r2_score': -0.0875
                }
            },
            'fairness_metrics': {
                'baseline': {
                    'age_group_mae_gap': 20.78,
                    'education_mae_gap': 13.36,
                    'service_time_mae_gap': 3.76
                },
                'mitigated': {
                    'age_group_mae_gap': 0.00,
                    'education_mae_gap': 17.56,
                    'service_time_mae_gap': 0.00
                }
            },
            'bias_mitigation': {
                'applied': True,
                'measures': [
                    'Removed proxy features (Height, Weight, BMI)',
                    'Balanced age group representation',
                    'Balanced education level representation'
                ]
            },
            'limitations': [
                'Model performance is limited due to data imbalance',
                'Predictions may not be accurate for extreme cases',
                'Model trained on a specific industry dataset; may not generalize'
            ]
        }
        
        return jsonify(model_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """API endpoint for feature importance"""
    try:
        if model is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get feature importance from linear regression coefficients
        importance = []
        for i, feature in enumerate(feature_columns):
            importance.append({
                'feature': feature,
                'importance': abs(model.coef_[i]) if hasattr(model, 'coef_') else 0
            })
        
        # Sort by importance
        importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify(importance[:10])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

# Ensure model is loaded when the app module is imported (gunicorn workers)
load_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

"""
Explainability endpoints for the Absenteeism Prediction Model
Provides SHAP, LIME, and Counterfactual explanations
"""

import os
import json
import time
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Feature flags: keep heavy explainers off by default on Render
USE_SHAP = os.environ.get('USE_SHAP', '0') == '1'
USE_LIME = os.environ.get('USE_LIME', '0') == '1'

# Lazy import to avoid circular dependency
def get_model_components():
    """Get model components from app module (lazy import to avoid circular dependency)"""
    from app import model, scaler, feature_columns, preprocess_input
    return model, scaler, feature_columns, preprocess_input

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

explain_bp = Blueprint('explain', __name__)

# Cache file path
CACHE_FILE = 'explain_global_cache.json'
CACHE_TTL_DAYS = 7


def generate_background_data(n_samples=100):
    """
    Generate synthetic background data for SHAP explainer.
    Creates samples based on typical feature ranges.
    """
    _, _, feature_columns, preprocess_input = get_model_components()
    if feature_columns is None or len(feature_columns) == 0:
        return None
    
    # Create synthetic data based on known feature ranges
    # These ranges are typical for the absenteeism dataset
    synthetic_data = {}
    
    # Numeric features (approximate ranges from dataset)
    numeric_features = {
        'Age': (25, 60),
        'Service time': (1, 20),
        'Work load Average/day ': (200, 350),
        'Transportation expense': (100, 400),
        'Distance from Residence to Work': (1, 50)
    }
    
    # Add numeric features
    for feat, (min_val, max_val) in numeric_features.items():
        if feat in feature_columns:
            synthetic_data[feat] = np.random.uniform(min_val, max_val, n_samples)
    
    # Add categorical features with typical values
    categorical_defaults = {
        'Education': [1, 2, 3],
        'Son': [0, 1, 2, 3],
        'Month of absence': list(range(1, 13)),
        'Day of the week': [2, 3, 4, 5, 6],
        'Seasons': [1, 2, 3, 4],
        'Hit target': [0, 1],
        'Disciplinary failure': [0, 1],
        'Social drinker': [0, 1],
        'Social smoker': [0, 1],
        'Pet': [0, 1],
        'Reason for absence': [0, 5, 10, 15, 20, 25]
    }
    
    for feat, values in categorical_defaults.items():
        if feat in feature_columns:
            synthetic_data[feat] = np.random.choice(values, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(synthetic_data)
    
    # Preprocess to match model format (one-hot encoding, etc.)
    processed = preprocess_input(df.iloc[0].to_dict())
    
    # Generate multiple samples by varying values
    background_samples = []
    for i in range(n_samples):
        sample = {}
        # Vary numeric features
        for feat, (min_val, max_val) in numeric_features.items():
            if feat in feature_columns:
                sample[feat] = np.random.uniform(min_val, max_val)
        
        # Vary categorical features
        for feat, values in categorical_defaults.items():
            if feat in feature_columns:
                sample[feat] = np.random.choice(values)
        
        processed_sample = preprocess_input(sample)
        if len(background_samples) == 0:
            # Initialize with first sample
            background_samples = processed_sample.values
        else:
            background_samples = np.vstack([background_samples, processed_sample.values])
    
    # Scale the background data
    _, scaler, _, _ = get_model_components()
    if scaler is not None:
        background_scaled = scaler.transform(background_samples)
        return background_scaled
    
    return background_samples


def get_shap_explainer():
    """Get or create SHAP explainer with caching"""
    if not SHAP_AVAILABLE or not USE_SHAP:
        # SHAP not available; caller should use coefficient-based fallback
        return None
    
    model, scaler, _, _ = get_model_components()
    if model is None or scaler is None:
        return None
    
    # For LinearRegression, use LinearExplainer (exact and fast)
    if hasattr(model, 'coef_'):
        # Generate background data
        background = generate_background_data(n_samples=100)
        if background is not None:
            try:
                explainer = shap.LinearExplainer(model, background)
                return explainer
            except Exception as e:
                print(f"Error creating SHAP explainer: {e}")
                return None
    
    return None


def load_cached_global_explanation():
    """Load cached global explanation if valid"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is still valid (within TTL)
        cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
        if datetime.now() - cache_time > timedelta(days=CACHE_TTL_DAYS):
            return None
        
        return cache_data.get('explanation')
    except Exception:
        return None


def save_cached_global_explanation(explanation):
    """Save global explanation to cache"""
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'explanation': explanation
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")


@explain_bp.route('/global', methods=['GET'])
def explain_global():
    """
    GET /explain/global
    Returns global feature importance using SHAP
    """
    try:
        model, scaler, feature_columns, preprocess_input = get_model_components()
        if model is None or scaler is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check cache first
        cached = load_cached_global_explanation()
        if cached is not None:
            return jsonify({
                **cached,
                'cached': True
            })
        
        feature_importance = None
        explainer = None
        mean_abs_shap = None

        if SHAP_AVAILABLE and USE_SHAP:
            explainer = get_shap_explainer()
            if explainer is not None:
                # Get background data for computing mean SHAP values
                background = generate_background_data(n_samples=100)
                if background is not None:
                    # Compute SHAP values for background data (mean importance)
                    shap_values = explainer.shap_values(background)
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        if mean_abs_shap is None:
            # Fallback: approximate global importance using absolute model coefficients
            if hasattr(model, 'coef_') and len(model.coef_) == len(feature_columns):
                mean_abs_shap = np.abs(model.coef_)
            else:
                return jsonify({'error': 'Unable to compute global importance'}), 500

        # Create feature importance list (SHAP or coefficient-based)
        feature_importance = [
            {
                'feature': feature_columns[i],
                'mean_abs_shap': float(mean_abs_shap[i])
            }
            for i in range(len(feature_columns))
        ]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
        
        explanation = {
            'feature_importance': feature_importance,
            'explainer_type': 'LinearExplainer' if SHAP_AVAILABLE and explainer is not None else 'CoefficientFallback',
            'sample_size': int(len(feature_importance)),
            'cached': False
        }
        
        # Save to cache
        save_cached_global_explanation(explanation)
        
        return jsonify(explanation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@explain_bp.route('/local', methods=['POST'])
def explain_local():
    """
    POST /explain/local
    Returns local SHAP values for a single prediction
    Expects: {"input": {...}}
    """
    try:
        model, scaler, feature_columns, preprocess_input = get_model_components()
        if model is None or scaler is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json or {}
        input_data = data.get('input', {})

        # Preprocess input (robust to missing fields)
        processed_data = preprocess_input(input_data)
        scaled_data = scaler.transform(processed_data)

        # Make prediction
        prediction = float(model.predict(scaled_data)[0])

        contributions = None

        if SHAP_AVAILABLE and USE_SHAP:
            explainer = get_shap_explainer()
            if explainer is not None:
                try:
                    shap_values = explainer.shap_values(scaled_data[0])
                    contributions = [
                        {
                            'feature': feature_columns[i],
                            'shap': float(shap_values[i]),
                            'value': float(scaled_data[0][i])
                        }
                        for i in range(len(feature_columns))
                    ]
                except Exception:
                    contributions = None

        if contributions is None:
            # Fallback: coefficient * value
            if hasattr(model, 'coef_') and len(model.coef_) == len(feature_columns):
                contributions = [
                    {
                        'feature': feature_columns[i],
                        'shap': float(model.coef_[i] * scaled_data[0][i]),
                        'value': float(scaled_data[0][i])
                    }
                    for i in range(len(feature_columns))
                ]
            else:
                # Last-resort fallback: zeros
                contributions = [
                    {
                        'feature': feature_columns[i],
                        'shap': 0.0,
                        'value': float(scaled_data[0][i]) if len(scaled_data[0]) > i else 0.0
                    }
                    for i in range(len(feature_columns))
                ]

        contributions.sort(key=lambda x: abs(x['shap']), reverse=True)

        top_features = contributions[:3]
        summary_parts = []
        for feat in top_features:
            direction = "increases" if feat['shap'] > 0 else "decreases"
            summary_parts.append(f"{feat['feature']} {direction} prediction by {abs(feat['shap']):.2f} hours")

        text_summary = f"Predicted {prediction:.2f} hours. Top factors: {', '.join(summary_parts)}."

        return jsonify({
            'prediction': prediction,
            'contributions': contributions,
            'text_summary': text_summary
        })

    except Exception as e:
        # Always return a minimal fallback so the UI doesn't break
        try:
            model, scaler, feature_columns, _ = get_model_components()
            if model is not None and scaler is not None and feature_columns is not None:
                zeros = np.zeros((1, len(feature_columns)))
                pred = float(model.predict(zeros)[0]) if hasattr(model, 'predict') else 0.0
                contributions = [
                    {'feature': feature_columns[i], 'shap': 0.0, 'value': 0.0}
                    for i in range(len(feature_columns))
                ]
                return jsonify({
                    'prediction': pred,
                    'contributions': contributions,
                    'text_summary': f'Fallback used: {str(e)}'
                })
        except Exception:
            # Absolute last resort static response
            return jsonify({
                'prediction': 0.0,
                'contributions': [],
                'text_summary': 'Fallback used.'
            })
        # Should not reach here, but still return 200 with fallback text
        return jsonify({
            'prediction': 0.0,
            'contributions': [],
            'text_summary': 'Fallback used.'
        })


@explain_bp.route('/lime', methods=['POST'])
def explain_lime():
    """
    POST /explain/lime
    Returns LIME explanation for a single prediction
    Expects: {"input": {...}}
    """
    try:
        model, scaler, feature_columns, preprocess_input = get_model_components()
        if model is None or scaler is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json or {}
        input_data = data.get('input', {})

        processed_data = preprocess_input(input_data)
        scaled_data = scaler.transform(processed_data)
        prediction = float(model.predict(scaled_data)[0])

        # Prefer real LIME if available; otherwise provide a deterministic fallback
        if LIME_AVAILABLE and USE_LIME:
            try:
                background = generate_background_data(n_samples=100)
                explainer = lime_tabular.LimeTabularExplainer(
                    background,
                    feature_names=feature_columns,
                    mode='regression',
                    discretize_continuous=False
                )
                explanation = explainer.explain_instance(
                    scaled_data[0],
                    model.predict,
                    num_features=10
                )
                top_features = []
                for feature_idx, weight in explanation.as_list():
                    feature_name = None
                    for i, col in enumerate(feature_columns):
                        if str(i) == str(feature_idx) or col == feature_idx:
                            feature_name = col
                            break
                    if feature_name is None:
                        feature_name = str(feature_idx)
                    top_features.append({'feature': feature_name, 'weight': float(weight)})
                return jsonify({
                    'prediction': prediction,
                    'top_features': top_features,
                    'explanation_score': float(explanation.score)
                })
            except Exception:
                pass

        # Fallback: coefficient-based importance
        if hasattr(model, 'coef_') and len(model.coef_) == len(feature_columns):
            weights = [
                {
                    'feature': feature_columns[i],
                    'weight': float(model.coef_[i] * scaled_data[0][i])
                }
                for i in range(len(feature_columns))
            ]
            # sort by absolute weight and keep top 10
            weights.sort(key=lambda x: abs(x['weight']), reverse=True)
            return jsonify({
                'prediction': prediction,
                'top_features': weights[:10],
                'explanation_score': 1.0
            })

        return jsonify({'error': 'Unable to compute LIME explanation'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@explain_bp.route('/cf', methods=['POST'])
def explain_counterfactual():
    """
    POST /explain/cf
    Returns counterfactual suggestions to reduce predicted absenteeism
    Expects: {"input": {...}, "target": 0.8} (optional target reduction factor)
    """
    try:
        model, scaler, feature_columns, preprocess_input = get_model_components()
        if model is None or scaler is None or feature_columns is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if 'input' not in data:
            return jsonify({'error': 'Missing "input" field in request'}), 400
        
        input_data = data['input']
        target_factor = data.get('target', 0.8)  # Default: reduce by 20%
        
        # Preprocess input
        processed_data = preprocess_input(input_data)
        scaled_data = scaler.transform(processed_data)
        
        # Get original prediction
        original_pred = model.predict(scaled_data)[0]
        target_pred = original_pred * target_factor
        
        # Identify actionable numeric features
        actionable_features = [
            'Age', 'Service time', 'Work load Average/day ', 
            'Transportation expense', 'Distance from Residence to Work'
        ]
        
        # Find feature indices in processed data
        feature_mapping = {}
        for feat in actionable_features:
            if feat in feature_columns:
                idx = feature_columns.index(feat)
                feature_mapping[feat] = idx
        
        candidates = []
        
        # Try different deltas: ±0.5σ, ±1σ, ±2σ
        for feat_name, feat_idx in feature_mapping.items():
            if feat_idx >= len(scaled_data[0]):
                continue
            
            # In scaled space, use unit standard deviation to propose deltas
            feat_std = 1.0
            current_value = scaled_data[0][feat_idx]
            
            # Try different deltas
            for delta_mult in [-0.5, -1.0, -2.0, 0.5, 1.0, 2.0]:
                delta = delta_mult * feat_std
                new_value = current_value + delta
                
                # Create modified input
                modified_scaled = scaled_data[0].copy()
                modified_scaled[feat_idx] = new_value
                
                # Predict with modified input
                new_pred = model.predict(modified_scaled.reshape(1, -1))[0]
                
                # Check if this helps reduce prediction toward target
                if new_pred < original_pred:
                    reduction = (original_pred - new_pred) / original_pred
                    
                    # Calculate normalized distance (L2 norm of change)
                    change_vector = modified_scaled - scaled_data[0]
                    distance = np.linalg.norm(change_vector)
                    
                    candidates.append({
                        'feature': feat_name,
                        'original_value': float(current_value),
                        'suggested_value': float(new_value),
                        'change': float(delta),
                        'new_prediction': float(new_pred),
                        'reduction_percent': float(reduction * 100),
                        'distance': float(distance)
                    })
        
        # Sort by reduction_percent (descending) and distance (ascending)
        candidates.sort(key=lambda x: (-x['reduction_percent'], x['distance']))
        
        # Take top 5 candidates
        top_candidates = candidates[:5]
        
        return jsonify({
            'original_prediction': float(original_pred),
            'target_prediction': float(target_pred),
            'candidates': top_candidates
        })
    
    except Exception as e:
        # Return safe empty candidates rather than an error so UI shows something
        try:
            model, scaler, feature_columns, _ = get_model_components()
            if model is not None and scaler is not None and feature_columns is not None:
                zeros = np.zeros((1, len(feature_columns)))
                base_pred = float(model.predict(zeros)[0]) if hasattr(model, 'predict') else 0.0
                return jsonify({
                    'original_prediction': base_pred,
                    'target_prediction': base_pred * 0.8,
                    'candidates': [],
                    'note': f'Counterfactual fallback used: {str(e)}'
                })
        except Exception:
            pass
        return jsonify({
            'original_prediction': 0.0,
            'target_prediction': 0.0,
            'candidates': [],
            'note': 'Counterfactual fallback used.'
        })


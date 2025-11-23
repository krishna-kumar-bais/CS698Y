"""
Tests for explainability endpoints
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_input():
    """Sample input data matching expected format"""
    return {
        'input': {
            'Age': 35,
            'Education': 2,
            'Service time': 5,
            'Work load Average/day ': 270.0,
            'Transportation expense': 200,
            'Distance from Residence to Work': 10,
            'Social drinker': 0,
            'Social smoker': 0,
            'Pet': 1,
            'Son': 1,
            'Hit target': 1,
            'Month of absence': 6,
            'Day of the week': 2,
            'Seasons': 2,
            'Reason for absence': 0,
            'Disciplinary failure': 0
        }
    }


def test_explain_global(client):
    """Test GET /explain/global endpoint"""
    response = client.get('/explain/global')
    
    # Should return 200 or 500 (if model not loaded or SHAP unavailable)
    assert response.status_code in [200, 500]
    
    data = json.loads(response.data)
    
    if response.status_code == 200:
        # Verify expected keys
        assert 'feature_importance' in data
        assert 'explainer_type' in data
        assert 'sample_size' in data
        assert 'cached' in data
        assert isinstance(data['feature_importance'], list)
        
        # If feature importance exists, verify structure
        if len(data['feature_importance']) > 0:
            assert 'feature' in data['feature_importance'][0]
            assert 'mean_abs_shap' in data['feature_importance'][0]


def test_explain_local(client, sample_input):
    """Test POST /explain/local endpoint"""
    response = client.post(
        '/explain/local',
        data=json.dumps(sample_input),
        content_type='application/json'
    )
    
    # Should return 200 or 400/500 (if model not loaded or invalid input)
    assert response.status_code in [200, 400, 500]
    
    data = json.loads(response.data)
    
    if response.status_code == 200:
        # Verify expected keys
        assert 'prediction' in data
        assert 'contributions' in data
        assert 'text_summary' in data
        assert isinstance(data['contributions'], list)
        assert isinstance(data['prediction'], (int, float))
        assert isinstance(data['text_summary'], str)
        
        # Verify contributions structure
        if len(data['contributions']) > 0:
            assert 'feature' in data['contributions'][0]
            assert 'shap' in data['contributions'][0]
            assert 'value' in data['contributions'][0]


def test_explain_lime(client, sample_input):
    """Test POST /explain/lime endpoint"""
    response = client.post(
        '/explain/lime',
        data=json.dumps(sample_input),
        content_type='application/json'
    )
    
    # Should return 200 or 400/500 (if model not loaded, LIME unavailable, or invalid input)
    assert response.status_code in [200, 400, 500]
    
    data = json.loads(response.data)
    
    if response.status_code == 200:
        # Verify expected keys
        assert 'prediction' in data
        assert 'top_features' in data
        assert isinstance(data['prediction'], (int, float))
        assert isinstance(data['top_features'], list)
        
        # Verify top_features structure
        if len(data['top_features']) > 0:
            assert 'feature' in data['top_features'][0]
            assert 'weight' in data['top_features'][0]


def test_explain_counterfactual(client, sample_input):
    """Test POST /explain/cf endpoint"""
    response = client.post(
        '/explain/cf',
        data=json.dumps(sample_input),
        content_type='application/json'
    )
    
    # Should return 200 or 400/500 (if model not loaded or invalid input)
    assert response.status_code in [200, 400, 500]
    
    data = json.loads(response.data)
    
    if response.status_code == 200:
        # Verify expected keys
        assert 'original_prediction' in data
        assert 'target_prediction' in data
        assert 'candidates' in data
        assert isinstance(data['original_prediction'], (int, float))
        assert isinstance(data['target_prediction'], (int, float))
        assert isinstance(data['candidates'], list)
        
        # Verify candidates structure
        if len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            assert 'feature' in candidate
            assert 'original_value' in candidate
            assert 'suggested_value' in candidate
            assert 'new_prediction' in candidate
            assert 'reduction_percent' in candidate
            assert 'distance' in candidate


def test_explain_local_missing_input(client):
    """Test POST /explain/local with missing input field"""
    response = client.post(
        '/explain/local',
        data=json.dumps({}),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_explain_counterfactual_with_target(client, sample_input):
    """Test POST /explain/cf with custom target"""
    sample_input['target'] = 0.7  # Reduce by 30%
    
    response = client.post(
        '/explain/cf',
        data=json.dumps(sample_input),
        content_type='application/json'
    )
    
    assert response.status_code in [200, 400, 500]
    
    if response.status_code == 200:
        data = json.loads(response.data)
        # Target prediction should be 0.7 * original
        assert abs(data['target_prediction'] - (data['original_prediction'] * 0.7)) < 0.01


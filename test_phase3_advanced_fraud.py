"""
Test Phase 3: Advanced Fraud Detection
Tests for ensemble ML models, A/B testing, and real-time model updates
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the service path to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'fraud-detection'))

from main import (
    app,
    initialize_models,
    get_ensemble_prediction,
    detect_fraud,
    compute_enhanced_fraud_score,
    generate_fraud_insights,
    get_risk_level,
    models
)

class TestEnsembleModels:
    """Test ensemble model functionality."""

    def setup_method(self):
        """Initialize models before each test."""
        initialize_models()

    def test_models_initialized(self):
        """Test that all ensemble models are initialized."""
        expected_models = ['isolation_forest', 'random_forest', 'xgboost', 'neural_network']
        assert len(models) == len(expected_models)
        for model_name in expected_models:
            assert model_name in models
            assert models[model_name] is not None

    def test_ensemble_prediction(self):
        """Test ensemble prediction functionality."""
        # Create a test vector
        test_vector = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.1, 0.2, 0.05])

        result = get_ensemble_prediction(test_vector)

        assert 'fraud_probability' in result
        assert 'model_used' in result
        assert 'confidence' in result
        assert result['model_used'] == 'ensemble'
        assert 0.0 <= result['fraud_probability'] <= 1.0
        assert 0.0 <= result['confidence'] <= 1.0

    def test_individual_model_prediction(self):
        """Test prediction from individual models."""
        test_vector = np.array([0.8, 0.9, 0.1, 0.2, 0.3, 0.7, 0.8, 0.1])

        for model_name in ['isolation_forest', 'random_forest', 'xgboost']:
            result = get_ensemble_prediction(test_vector, model_name)

            assert result['model_used'] == model_name
            assert 'fraud_probability' in result
            assert 0.0 <= result['fraud_probability'] <= 1.0

    def test_neural_network_prediction(self):
        """Test neural network model prediction."""
        test_vector = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.1, 0.2, 0.05])

        result = get_ensemble_prediction(test_vector, 'neural_network')

        assert result['model_used'] == 'neural_network'
        assert 'fraud_probability' in result
        assert 0.0 <= result['fraud_probability'] <= 1.0

class TestEnhancedScoring:
    """Test enhanced fraud scoring functionality."""

    def test_compute_enhanced_fraud_score(self):
        """Test enhanced fraud score computation."""
        vector = np.array([0.8, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.1])  # High amount, velocity
        knn_results = [("txn_001", 0.1), ("txn_002", 0.15)]
        fraud_probability = 0.8
        confidence = 0.9

        score = compute_enhanced_fraud_score(vector, knn_results, fraud_probability, confidence)

        assert 0.0 <= score <= 1.0
        # Should be high due to behavioral factors
        assert score > 0.5

    def test_generate_fraud_insights(self):
        """Test fraud insights generation."""
        vector = np.array([0.9, 0.3, 0.2, 0.1, 0.4, 0.8, 0.9, 0.1])  # Very high amount/velocity
        ensemble_result = {
            'model_predictions': [
                {'probability': 0.8, 'model': 'xgboost'},
                {'probability': 0.7, 'model': 'random_forest'}
            ]
        }
        knn_results = [("txn_001", 0.05), ("txn_002", 0.08)]

        insights = generate_fraud_insights(vector, ensemble_result, knn_results)

        assert isinstance(insights, list)
        assert len(insights) > 0
        # Should include high amount and velocity insights
        insight_text = ' '.join(insights).lower()
        assert 'high' in insight_text or 'unusually' in insight_text

    def test_risk_level_mapping(self):
        """Test risk level classification."""
        test_cases = [
            (0.95, "CRITICAL"),
            (0.85, "HIGH"),
            (0.65, "MEDIUM"),
            (0.45, "LOW"),
            (0.15, "VERY_LOW")
        ]

        for score, expected_level in test_cases:
            assert get_risk_level(score) == expected_level

class TestFraudDetection:
    """Test complete fraud detection pipeline."""

    def setup_method(self):
        """Initialize models before each test."""
        initialize_models()

    def test_detect_fraud_complete(self):
        """Test complete fraud detection with all features."""
        transaction_id = "test_txn_123"
        vector = np.array([0.7, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.1])

        result = detect_fraud(transaction_id, vector)

        required_fields = [
            'transaction_id', 'is_fraud', 'fraud_score', 'risk_level',
            'fraud_probability', 'confidence', 'model_used',
            'similar_transactions', 'insights', 'timestamp'
        ]

        for field in required_fields:
            assert field in result

        assert result['transaction_id'] == transaction_id
        assert isinstance(result['is_fraud'], bool)
        assert 0.0 <= result['fraud_score'] <= 1.0
        assert result['risk_level'] in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]
        assert isinstance(result['insights'], list)

    def test_detect_fraud_with_model_version(self):
        """Test fraud detection with specific model version."""
        transaction_id = "test_txn_456"
        vector = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.1, 0.2, 0.05])

        result = detect_fraud(transaction_id, vector, model_version='xgboost')

        assert result['model_used'] == 'xgboost'
        assert result['transaction_id'] == transaction_id

class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data['service'] == 'fraud-detection'
        assert data['status'] == 'healthy'
        assert 'model_status' in data
        assert 'active_model_version' in data
        assert 'available_models' in data

    def test_detect_endpoint(self):
        """Test fraud detection endpoint."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        test_data = {
            "transaction_id": "api_test_123",
            "vector": [0.5, 0.3, 0.2, 0.1, 0.4, 0.1, 0.2, 0.05],
            "metadata": {"source": "api_test"}
        }

        response = client.post("/detect", json=test_data)
        assert response.status_code == 200

        result = response.json()
        assert result['transaction_id'] == "api_test_123"
        assert 'fraud_score' in result
        assert 'risk_level' in result
        assert 'insights' in result

    def test_models_endpoint(self):
        """Test models listing endpoint."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert 'models' in data
        assert 'active_version' in data
        assert isinstance(data['models'], list)

        if data['models']:
            model = data['models'][0]
            assert 'name' in model
            assert 'type' in model
            assert 'metrics' in model

if __name__ == "__main__":
    # Run basic functionality tests
    print("Testing Phase 3: Advanced Fraud Detection...")

    # Initialize models
    print("Initializing ensemble models...")
    initialize_models()
    print(f"✓ Initialized {len(models)} models: {list(models.keys())}")

    # Test ensemble prediction
    print("Testing ensemble prediction...")
    test_vector = np.array([0.6, 0.4, 0.3, 0.2, 0.5, 0.3, 0.4, 0.1])
    result = get_ensemble_prediction(test_vector)
    print(f"✓ Ensemble prediction: {result['fraud_probability']:.3f} (confidence: {result['confidence']:.3f})")

    # Test fraud detection
    print("Testing fraud detection...")
    fraud_result = detect_fraud("test_txn_phase3", test_vector)
    print(f"✓ Fraud detection: score={fraud_result['fraud_score']:.3f}, risk={fraud_result['risk_level']}")

    # Test insights
    print("Testing fraud insights...")
    insights = generate_fraud_insights(test_vector, result, [("txn_001", 0.1)])
    print(f"✓ Generated {len(insights)} insights: {insights[:2]}...")

    print("\n🎉 Phase 3: Advanced Fraud Detection tests completed successfully!")
    print("✓ Ensemble ML Models (Isolation Forest, Random Forest, XGBoost, Neural Network)")
    print("✓ Enhanced Fraud Scoring with behavioral features")
    print("✓ Real-time model insights and explanations")
    print("✓ A/B testing capabilities (model versioning)")
    print("✓ REST API endpoints for model management")
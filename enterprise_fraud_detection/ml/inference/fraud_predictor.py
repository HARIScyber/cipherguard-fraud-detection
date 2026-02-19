"""
Advanced Model Inference System with Ensemble Prediction and Explainability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import joblib
import json
import logging
from pathlib import Path
from dataclasses import dataclass
import time

# ML libraries for inference
import shap
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from ml.features.feature_engineering import AdvancedFeatureExtractor, FeatureVector

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for fraud prediction results."""
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    confidence: float
    risk_level: str
    individual_scores: Dict[str, float]
    risk_factors: List[str]
    shap_values: Optional[Dict[str, float]]
    processing_time_ms: float
    model_version: str
    feature_vector: Optional[List[float]] = None


@dataclass
class ModelInfo:
    """Container for model metadata."""
    model_type: str
    model_version: str
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: str


class FraudDetectionPredictor:
    """Advanced fraud detection with ensemble prediction and explainability."""
    
    def __init__(self, models_dir: str = None, db_session = None):
        self.models_dir = Path(models_dir or settings.ml.models_dir)
        self.db_session = db_session
        
        # Loaded models
        self.models = {}
        self.model_metadata = {}
        self.ensemble_weights = settings.ml.ensemble_weights
        
        # Feature extractor
        if db_session:
            self.feature_extractor = AdvancedFeatureExtractor(db_session)
        else:
            self.feature_extractor = None
        
        # SHAP explainers for interpretability
        self.explainers = {}
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': settings.ml.low_risk_threshold,
            'medium': settings.ml.medium_risk_threshold,
            'high': settings.ml.high_risk_threshold
        }
        
        # Load models on initialization
        self._load_models()
    
    def _load_models(self):
        """Load all available trained models."""
        logger.info("Loading trained models...")
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Find model files
        model_files = list(self.models_dir.glob("*.joblib"))
        
        for model_file in model_files:
            try:
                # Load model
                model = joblib.load(model_file)
                
                # Load metadata
                metadata_file = model_file.with_suffix('').with_suffix('_metadata.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    model_type = metadata['model_type']
                    model_version = metadata['model_version']
                    
                    self.models[model_type] = model
                    self.model_metadata[model_type] = ModelInfo(
                        model_type=model_type,
                        model_version=model_version,
                        feature_names=metadata['feature_names'],
                        hyperparameters=metadata['hyperparameters'],
                        performance_metrics=metadata['metrics'],
                        created_at=metadata['created_at']
                    )
                    
                    logger.info(f"Loaded {model_type} model (v{model_version})")
                    
                    # Initialize SHAP explainer
                    self._initialize_shap_explainer(model_type, model)
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def _initialize_shap_explainer(self, model_type: str, model: Any):
        """Initialize SHAP explainer for model interpretability."""
        try:
            if model_type == 'isolation_forest':
                # TreeExplainer for tree-based models
                self.explainers[model_type] = shap.Explainer(model)
            elif model_type in ['random_forest', 'xgboost']:
                # For pipeline models, extract the actual estimator
                if hasattr(model, 'named_steps'):
                    actual_model = model.named_steps['model']
                else:
                    actual_model = model
                self.explainers[model_type] = shap.TreeExplainer(actual_model)
            elif model_type == 'ensemble':
                # For ensemble, we'll use the first available explainer
                pass  # Will be handled separately
            
            logger.debug(f"SHAP explainer initialized for {model_type}")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer for {model_type}: {e}")
    
    def predict_fraud(
        self, 
        transaction_data: Dict[str, Any],
        include_explanations: bool = True,
        include_feature_vector: bool = False
    ) -> PredictionResult:
        """
        Predict fraud probability for a transaction using ensemble of models.
        
        Args:
            transaction_data: Raw transaction data
            include_explanations: Whether to include SHAP explanations
            include_feature_vector: Whether to include feature vector in result
            
        Returns:
            PredictionResult with fraud score and explanations
        """
        start_time = time.time()
        transaction_id = transaction_data.get('transaction_id', f'txn_{int(time.time())}')
        
        try:
            # Extract features
            if self.feature_extractor:
                # Use advanced feature extractor with database context
                feature_vector = self.feature_extractor.extract_features(transaction_data)
                features = feature_vector.features
                feature_names = feature_vector.feature_names
            else:
                # Use simplified feature extraction
                features, feature_names = self._extract_simple_features(transaction_data)
            
            # Make predictions with individual models
            individual_scores = {}
            individual_confidences = {}
            
            # Ensemble prediction
            ensemble_scores = []
            ensemble_weights = []
            
            for model_type, model in self.models.items():
                if model_type == 'isolation_forest':
                    score, confidence = self._predict_isolation_forest(model, features)
                elif model_type in ['random_forest', 'xgboost']:
                    score, confidence = self._predict_supervised_model(model, features)
                elif model_type == 'ensemble':
                    score, confidence = self._predict_supervised_model(model, features)
                else:
                    continue
                
                individual_scores[model_type] = score
                individual_confidences[model_type] = confidence
                
                # Add to ensemble (exclude ensemble model from ensemble calculation)
                if model_type != 'ensemble':
                    ensemble_scores.append(score)
                    # Use model performance as weight
                    weight = self.model_metadata[model_type].performance_metrics.get('f1_score', 1.0)
                    ensemble_weights.append(weight)
            
            # Calculate ensemble fraud score
            if ensemble_scores:
                # Weighted average
                ensemble_weights = np.array(ensemble_weights)
                ensemble_weights = ensemble_weights / np.sum(ensemble_weights)  # Normalize
                fraud_score = np.average(ensemble_scores, weights=ensemble_weights)
                
                # Calculate ensemble confidence
                confidence = np.average(list(individual_confidences.values()))
            else:
                # Fallback if no models available
                fraud_score = 0.5
                confidence = 0.0
                individual_scores = {'fallback': 0.5}
            
            # Determine risk level and fraud flag
            is_fraud, risk_level = self._determine_risk_level(fraud_score)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(transaction_data, fraud_score, features)
            
            # Generate explanations if requested
            shap_values = None
            if include_explanations and self.explainers:
                shap_values = self._generate_explanations(features, feature_names)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Get model version info
            model_versions = [info.model_version for info in self.model_metadata.values()]
            model_version = f"ensemble_{'+'.join(model_versions)}" if len(model_versions) > 1 else model_versions[0] if model_versions else "unknown"
            
            # Create result
            result = PredictionResult(
                transaction_id=transaction_id,
                fraud_score=float(fraud_score),
                is_fraud=is_fraud,
                confidence=float(confidence),
                risk_level=risk_level,
                individual_scores=individual_scores,
                risk_factors=risk_factors,
                shap_values=shap_values,
                processing_time_ms=processing_time_ms,
                model_version=model_version,
                feature_vector=features.tolist() if include_feature_vector else None
            )
            
            logger.debug(
                f"Prediction completed for {transaction_id}: "
                f"score={fraud_score:.3f}, risk={risk_level}, time={processing_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {transaction_id}: {e}")
            
            # Return safe fallback result
            processing_time_ms = (time.time() - start_time) * 1000
            return PredictionResult(
                transaction_id=transaction_id,
                fraud_score=0.5,
                is_fraud=False,
                confidence=0.0,
                risk_level='UNKNOWN',
                individual_scores={'error': 0.5},
                risk_factors=['prediction_error'],
                shap_values=None,
                processing_time_ms=processing_time_ms,
                model_version='error'
            )
    
    def _predict_isolation_forest(self, model: Any, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction with Isolation Forest model."""
        try:
            # Reshape for single sample
            features_2d = features.reshape(1, -1)
            
            # Get anomaly score
            anomaly_score = model.decision_function(features_2d)[0]
            
            # Convert to fraud probability (0-1 range)
            # Isolation Forest: negative scores indicate anomalies
            fraud_score = max(0.0, min(1.0, (0.5 - anomaly_score)))
            
            # Confidence based on absolute score
            confidence = min(1.0, abs(anomaly_score) / 2.0)
            
            return fraud_score, confidence
            
        except Exception as e:
            logger.warning(f"Isolation Forest prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_supervised_model(self, model: Any, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction with supervised model (RandomForest, XGBoost, Ensemble)."""
        try:
            # Reshape for single sample
            features_2d = features.reshape(1, -1)
            
            # Get probability prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_2d)[0]
                fraud_score = proba[1]  # Probability of fraud class
                
                # Confidence based on how far from 0.5 the prediction is
                confidence = abs(fraud_score - 0.5) * 2
            else:
                # Binary prediction only
                pred = model.predict(features_2d)[0]
                fraud_score = float(pred)
                confidence = 1.0 if pred in [0, 1] else 0.0
            
            return fraud_score, confidence
            
        except Exception as e:
            logger.warning(f"Supervised model prediction failed: {e}")
            return 0.5, 0.0
    
    def _determine_risk_level(self, fraud_score: float) -> Tuple[bool, str]:
        """Determine risk level and fraud flag based on score."""
        
        is_fraud = fraud_score > self.risk_thresholds['medium']
        
        if fraud_score < self.risk_thresholds['low']:
            risk_level = 'LOW'
        elif fraud_score < self.risk_thresholds['medium']:
            risk_level = 'MEDIUM'
        elif fraud_score < self.risk_thresholds['high']:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'
        
        return is_fraud, risk_level
    
    def _identify_risk_factors(
        self, 
        transaction_data: Dict[str, Any], 
        fraud_score: float,
        features: np.ndarray
    ) -> List[str]:
        """Identify specific risk factors contributing to the fraud score."""
        
        risk_factors = []
        
        # High amount
        amount = float(transaction_data.get('amount', 0))
        if amount > 10000:
            risk_factors.append('High transaction amount')
        elif amount < 1:
            risk_factors.append('Unusually low amount')
        
        # Unknown or high-risk merchant
        merchant = transaction_data.get('merchant', '')
        if merchant.lower() in ['unknown', 'test_merchant', '']:
            risk_factors.append('Unknown or suspicious merchant')
        
        # Geographic risk
        country = transaction_data.get('country', '')
        if country in ['XX', 'Unknown', '']:
            risk_factors.append('High-risk or unknown location')
        
        # Time-based patterns
        timestamp = transaction_data.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                hour = dt.hour
                if hour < 6 or hour > 22:
                    risk_factors.append('Off-hours transaction timing')
                
                if dt.weekday() >= 5:  # Weekend
                    risk_factors.append('Weekend transaction')
            except:
                pass
        
        # High fraud score
        if fraud_score > 0.8:
            risk_factors.append('Anomalous transaction pattern detected')
        elif fraud_score > 0.6:
            risk_factors.append('Suspicious transaction characteristics')
        
        # Device and session risk
        device = transaction_data.get('device', '')
        if device == 'unknown':
            risk_factors.append('Unknown device type')
        
        return risk_factors
    
    def _generate_explanations(
        self, 
        features: np.ndarray, 
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Generate SHAP explanations for the prediction."""
        
        try:
            # Use the best available explainer
            explainer = None
            if 'ensemble' in self.explainers:
                explainer = self.explainers['ensemble']
            elif 'random_forest' in self.explainers:
                explainer = self.explainers['random_forest']
            elif 'xgboost' in self.explainers:
                explainer = self.explainers['xgboost']
            
            if explainer:
                # Generate SHAP values
                shap_values = explainer.shap_values(features.reshape(1, -1))
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Multi-class output, take fraud class (class 1)
                    shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    shap_vals = shap_values[0]
                
                # Create feature importance dictionary
                feature_importance = {}
                for name, importance in zip(feature_names, shap_vals):
                    feature_importance[name] = float(importance)
                
                # Sort by absolute importance
                sorted_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                )
                
                # Return top 10 most important features
                return dict(list(sorted_importance.items())[:10])
            
        except Exception as e:
            logger.warning(f"SHAP explanation generation failed: {e}")
        
        return None
    
    def _extract_simple_features(self, transaction_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Simplified feature extraction when advanced extractor is not available."""
        import math
        
        amount = float(transaction_data.get('amount', 0))
        merchant = transaction_data.get('merchant', 'Unknown')
        device = transaction_data.get('device', 'unknown')
        country = transaction_data.get('country', 'Unknown')
        
        # Simple merchant risk mapping
        merchant_risks = {
            'Unknown': 0.8, 'Test_Merchant': 0.6, 'Amazon': 0.1, 'Walmart': 0.15,
            'Apple': 0.12, 'Google': 0.18, 'Target': 0.2, 'Best Buy': 0.25,
            'Netflix': 0.1, 'Spotify': 0.1, 'Uber': 0.3, 'DoorDash': 0.25
        }
        
        # Extract timestamp features
        timestamp = transaction_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
        else:
            hour = 12  # Default to noon
            is_weekend = False
        
        features = [
            math.log1p(amount),                              # amount_log
            min(amount / 10000.0, 1.0),                    # amount_normalized
            hash(merchant) % 100 / 100.0,                  # merchant_hash
            {'mobile': 0.0, 'desktop': 0.5, 'tablet': 0.3}.get(device, 0.8),  # device_encoded
            hash(country) % 50 / 50.0,                     # country_hash
            1.0 if amount > 5000 else 0.0,                 # amount_high_flag 
            1.0 if amount < 10 else 0.0,                   # amount_low_flag
            merchant_risks.get(merchant, 0.5),             # merchant_risk
            hour / 24.0,                                   # hour_normalized
            1.0 if is_weekend else 0.0,                   # weekend_flag
        ]
        
        feature_names = [
            'amount_log', 'amount_normalized', 'merchant_hash', 'device_encoded',
            'country_hash', 'amount_high_flag', 'amount_low_flag', 'merchant_risk',
            'hour_normalized', 'weekend_flag'
        ]
        
        return np.array(features, dtype=np.float32), feature_names
    
    def get_model_info(self) -> Dict[str, ModelInfo]:
        """Get information about loaded models."""
        return self.model_metadata.copy()
    
    def is_ready(self) -> bool:
        """Check if predictor is ready with at least one model loaded."""
        return len(self.models) > 0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the predictor."""
        return {
            'is_ready': self.is_ready(),
            'models_loaded': list(self.models.keys()),
            'model_count': len(self.models),
            'feature_extractor_available': self.feature_extractor is not None,
            'explainers_available': list(self.explainers.keys()),
            'risk_thresholds': self.risk_thresholds
        }
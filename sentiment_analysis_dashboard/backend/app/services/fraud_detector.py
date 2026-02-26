"""
Fraud Detection Service
Machine learning-based fraud detection using Isolation Forest algorithm.
"""

import logging
import time
import os
import math
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Fraud detection service using Isolation Forest for anomaly detection.
    """
    
    VECTOR_DIM = 6  # Feature vector dimension
    
    # Feature mappings
    MERCHANTS = {
        "amazon": 0.1, "walmart": 0.2, "apple": 0.3, "google": 0.4,
        "target": 0.15, "bestbuy": 0.25, "ebay": 0.35, "paypal": 0.45
    }
    DEVICES = {"mobile": 0.1, "desktop": 0.5, "tablet": 0.3}
    COUNTRIES = {
        "us": 0.1, "uk": 0.2, "ca": 0.15, "au": 0.25, "de": 0.3, "fr": 0.35
    }
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        "CRITICAL": 0.9,
        "HIGH": 0.7,
        "MEDIUM": 0.5,
        "LOW": 0.3,
        "VERY_LOW": 0.0
    }
    
    def __init__(self):
        """Initialize the fraud detector."""
        self.model = None
        self.model_version = "isolation_forest_v1"
        self._is_loaded = False
    
    async def load_model(self):
        """Load the fraud detection model."""
        try:
            # Try to load sklearn model if available
            try:
                import joblib
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "..", "..", "models", "isolation_forest.joblib"
                )
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    logger.info(f"Loaded Isolation Forest model from {model_path}")
                    self._is_loaded = True
                else:
                    logger.warning("Model file not found, using rule-based detection")
                    self._is_loaded = True  # Use fallback
            except Exception as e:
                logger.warning(f"Could not load sklearn model: {e}, using rule-based detection")
                self._is_loaded = True  # Use fallback
                
        except Exception as e:
            logger.error(f"Failed to initialize fraud detector: {e}")
            self._is_loaded = True  # Still allow rule-based detection
    
    def load_model_sync(self):
        """Synchronous model loading for non-async contexts."""
        try:
            import joblib
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "..", "..", "models", "isolation_forest.joblib"
            )
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Loaded Isolation Forest model from {model_path}")
            else:
                logger.warning("Model file not found, using rule-based detection")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        self._is_loaded = True
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    def extract_features(self, transaction: Dict) -> np.ndarray:
        """
        Extract normalized features from a transaction.
        
        Args:
            transaction: Dict with keys: amount, merchant, device, country
        
        Returns:
            np.ndarray: Normalized 6-dimensional feature vector
        """
        features = []
        
        # 1. Amount (log-normalized)
        amount = float(transaction.get("amount", 0))
        amount_norm = self._normalize_amount(amount)
        features.append(amount_norm)
        
        # 2. Time-of-day embedding
        time_embedding = datetime.now().hour / 24.0
        features.append(time_embedding)
        
        # 3. Merchant embedding
        merchant = transaction.get("merchant", "other").lower()
        merchant_emb = self.MERCHANTS.get(merchant, 0.5)
        features.append(merchant_emb)
        
        # 4. Device fingerprint
        device = transaction.get("device", "desktop").lower()
        device_emb = self.DEVICES.get(device, 0.3)
        features.append(device_emb)
        
        # 5. Country embedding
        country = transaction.get("country", "us").lower()
        country_emb = self.COUNTRIES.get(country, 0.4)
        features.append(country_emb)
        
        # 6. Risk flag (unusual patterns)
        risk_flag = self._compute_risk_flag(transaction)
        features.append(risk_flag)
        
        # Convert to numpy array and normalize
        vector = np.array(features, dtype=np.float32)
        vector = self._normalize_vector(vector)
        
        return vector
    
    def _normalize_amount(self, amount: float) -> float:
        """Normalize transaction amount using log scale."""
        if amount <= 0:
            return 0.0
        log_amount = math.log(amount + 1)
        normalized = min(log_amount / 10.0, 1.0)
        return normalized
    
    def _compute_risk_flag(self, transaction: Dict) -> float:
        """Compute risk flag based on transaction patterns."""
        risk = 0.0
        
        # High amount risk
        amount = float(transaction.get("amount", 0))
        if amount > 5000:
            risk += 0.3
        elif amount > 1000:
            risk += 0.15
        
        # Unusual device
        device = transaction.get("device", "desktop").lower()
        if device == "mobile":
            risk += 0.05
        
        # Risky country
        country = transaction.get("country", "US").lower()
        safe_countries = ["us", "uk", "ca", "au", "de", "fr"]
        if country not in safe_countries:
            risk += 0.2
        
        # Unknown merchant risk
        merchant = transaction.get("merchant", "").lower()
        if merchant not in self.MERCHANTS:
            risk += 0.1
        
        return min(risk, 1.0)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize the feature vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def detect_fraud(self, transaction: Dict) -> Tuple[bool, float, str]:
        """
        Detect if a transaction is fraudulent.
        
        Args:
            transaction: Transaction details
        
        Returns:
            Tuple of (is_fraud, fraud_score, risk_level)
        """
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(transaction)
        
        # Get fraud score
        if self.model is not None:
            try:
                # Use Isolation Forest prediction
                # -1 for anomaly, 1 for normal
                prediction = self.model.predict(features.reshape(1, -1))[0]
                # Get anomaly score (higher = more anomalous)
                score = -self.model.score_samples(features.reshape(1, -1))[0]
                # Normalize to [0, 1]
                fraud_score = min(max((score + 0.5) / 1.0, 0.0), 1.0)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}, using rule-based")
                fraud_score = self._rule_based_score(transaction, features)
        else:
            # Rule-based detection
            fraud_score = self._rule_based_score(transaction, features)
        
        # Determine risk level
        risk_level = self._get_risk_level(fraud_score)
        
        # Is fraud based on threshold
        is_fraud = fraud_score >= 0.5
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Fraud detection completed in {processing_time:.2f}ms")
        
        return is_fraud, fraud_score, risk_level
    
    def _rule_based_score(self, transaction: Dict, features: np.ndarray) -> float:
        """
        Calculate fraud score using rule-based detection.
        """
        score = 0.0
        amount = float(transaction.get("amount", 0))
        
        # Amount-based scoring
        if amount > 10000:
            score += 0.4
        elif amount > 5000:
            score += 0.3
        elif amount > 2000:
            score += 0.2
        elif amount > 1000:
            score += 0.1
        
        # Country risk
        country = transaction.get("country", "US").lower()
        high_risk_countries = ["ru", "cn", "ng", "pk", "ua", "by"]
        medium_risk_countries = ["br", "mx", "in", "ph"]
        
        if country in high_risk_countries:
            score += 0.3
        elif country in medium_risk_countries:
            score += 0.15
        elif country not in ["us", "uk", "ca", "au", "de", "fr"]:
            score += 0.1
        
        # Unknown merchant
        merchant = transaction.get("merchant", "").lower()
        if merchant not in self.MERCHANTS:
            if "unknown" in merchant or len(merchant) < 3:
                score += 0.2
            else:
                score += 0.05
        
        # Device type
        device = transaction.get("device", "desktop").lower()
        if device == "mobile":
            score += 0.05
        
        # Time of day (late night transactions)
        hour = datetime.now().hour
        if hour >= 23 or hour <= 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on fraud score."""
        if score >= self.RISK_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif score >= self.RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif score >= self.RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif score >= self.RISK_THRESHOLDS["LOW"]:
            return "LOW"
        else:
            return "VERY_LOW"


# Singleton instance
fraud_detector = FraudDetector()


def get_fraud_detector() -> FraudDetector:
    """Get the fraud detector instance."""
    return fraud_detector

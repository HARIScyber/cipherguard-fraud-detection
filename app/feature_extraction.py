"""
Feature Extraction Module
Converts raw transactions into normalized vector embeddings
"""

import numpy as np
import hashlib
from typing import Dict, List
from datetime import datetime
import math


class FeatureExtractor:
    """
    Extracts and normalizes transaction features into a numerical vector.
    
    Features extracted:
    - Amount (log-normalized)
    - Time-of-day embedding
    - Merchant hash embedding
    - Device fingerprint
    - Country embedding
    - Risk flags
    """
    
    VECTOR_DIM = 6  # Output vector dimension
    MERCHANTS = {
        "Amazon": 0.1,
        "Walmart": 0.2,
        "Apple": 0.3,
        "Google": 0.4,
        "Other": 0.5
    }
    DEVICES = {"mobile": 0.1, "desktop": 0.5, "tablet": 0.3}
    COUNTRIES = {"US": 0.1, "UK": 0.2, "CA": 0.15, "AU": 0.25, "Other": 0.4}
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.merchant_mapping = self.MERCHANTS
        self.device_mapping = self.DEVICES
        self.country_mapping = self.COUNTRIES
    
    def extract_features(self, transaction: Dict) -> np.ndarray:
        """
        Extract normalized features from a transaction.
        
        Args:
            transaction: Dict with keys: amount, merchant, device, country, timestamp
        
        Returns:
            np.ndarray: Normalized 6-dimensional feature vector
        """
        features = []
        
        # 1. Amount (log-normalized)
        amount = float(transaction.get("amount", 0))
        amount_norm = self._normalize_amount(amount)
        features.append(amount_norm)
        
        # 2. Time-of-day embedding
        time_embedding = self._time_of_day_embedding(transaction.get("timestamp"))
        features.append(time_embedding)
        
        # 3. Merchant embedding
        merchant = transaction.get("merchant", "Other")
        merchant_emb = self.merchant_mapping.get(merchant, 0.5)
        features.append(merchant_emb)
        
        # 4. Device fingerprint
        device = transaction.get("device", "desktop")
        device_emb = self.device_mapping.get(device, 0.3)
        features.append(device_emb)
        
        # 5. Country embedding
        country = transaction.get("country", "Other")
        country_emb = self.country_mapping.get(country, 0.4)
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
        # Log scale: map to [0, 1]
        log_amount = math.log(amount + 1)
        normalized = min(log_amount / 10.0, 1.0)  # Cap at 1.0
        return normalized
    
    def _time_of_day_embedding(self, timestamp: str = None) -> float:
        """Create time-of-day embedding."""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                hour = dt.hour
            except:
                hour = datetime.now().hour
        else:
            hour = datetime.now().hour
        
        # Map hour to [0, 1]: midnight=0, noon=0.5, 11pm=0.95
        return hour / 24.0
    
    def _compute_risk_flag(self, transaction: Dict) -> float:
        """Compute risk flag based on transaction patterns."""
        risk = 0.0
        
        # High amount risk
        amount = float(transaction.get("amount", 0))
        if amount > 5000:
            risk += 0.3
        
        # Unusual device
        device = transaction.get("device", "desktop")
        if device == "mobile":
            risk += 0.1
        
        # Risky country
        country = transaction.get("country", "US")
        if country not in ["US", "UK", "CA"]:
            risk += 0.15
        
        return min(risk, 1.0)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize the feature vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


# Singleton instance
feature_extractor = FeatureExtractor()


def extract_transaction_vector(transaction: Dict) -> np.ndarray:
    """
    Convenience function to extract transaction vector.
    
    Args:
        transaction: Raw transaction dictionary
    
    Returns:
        np.ndarray: Normalized 6-dimensional vector
    """
    return feature_extractor.extract_features(transaction)

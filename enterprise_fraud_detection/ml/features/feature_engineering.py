"""
Advanced Feature Engineering for Fraud Detection
Comprehensive feature extraction including transaction patterns, user behavior, and contextual features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import math
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session

from database.repositories.transaction_repository import (
    TransactionRepository, UserBehaviorRepository
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Container for extracted features with metadata."""
    features: np.ndarray
    feature_names: List[str]
    transaction_id: str
    processing_time_ms: float
    feature_metadata: Dict[str, Any]


class AdvancedFeatureExtractor:
    """Advanced feature extraction with multiple feature categories."""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.transaction_repo = TransactionRepository(db_session)
        self.behavior_repo = UserBehaviorRepository(db_session)
        
        # Feature scalers and encoders
        self.amount_scaler = StandardScaler()
        self.country_encoder = LabelEncoder()
        self.merchant_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        
        # Merchant risk scores (would be computed from historical data)
        self.merchant_risk_scores = self._load_merchant_risk_scores()
        
        # Feature categories
        self.feature_categories = {
            'transaction_basic': 8,      # Basic transaction features
            'user_behavior': 10,         # User behavior patterns
            'velocity_features': 6,      # Transaction velocity
            'contextual_features': 8,    # Time, location, device context
            'merchant_features': 5,      # Merchant-specific features
            'network_features': 4        # IP and session features
        }
        
        self.total_features = sum(self.feature_categories.values())
        
    def _load_merchant_risk_scores(self) -> Dict[str, float]:
        """Load merchant risk scores from historical data."""
        # In production, this would load from database or external service
        return {
            'Amazon': 0.1,
            'Walmart': 0.15,
            'Apple': 0.12,
            'Google': 0.18,
            'Unknown': 0.8,
            'Test_Merchant': 0.5,
            'High_Risk_Merchant': 0.9,
            'Low_Risk_Merchant': 0.05
        }
    
    def extract_features(
        self, 
        transaction_data: Dict[str, Any], 
        include_metadata: bool = True
    ) -> FeatureVector:
        """
        Extract comprehensive feature vector from transaction data.
        
        Args:
            transaction_data: Raw transaction data
            include_metadata: Whether to include feature metadata
            
        Returns:
            FeatureVector with extracted features and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Initialize feature arrays for each category
            features = {}
            metadata = {}
            
            # 1. Basic transaction features
            basic_features, basic_metadata = self._extract_basic_features(transaction_data)
            features['transaction_basic'] = basic_features
            metadata['transaction_basic'] = basic_metadata
            
            # 2. User behavior features (if customer_id available)
            if transaction_data.get('customer_id'):
                behavior_features, behavior_metadata = self._extract_user_behavior_features(
                    transaction_data['customer_id']
                )
                features['user_behavior'] = behavior_features
                metadata['user_behavior'] = behavior_metadata
            else:
                features['user_behavior'] = np.zeros(self.feature_categories['user_behavior'])
                metadata['user_behavior'] = {'status': 'no_customer_id'}
            
            # 3. Velocity features
            velocity_features, velocity_metadata = self._extract_velocity_features(transaction_data)
            features['velocity_features'] = velocity_features
            metadata['velocity_features'] = velocity_metadata
            
            # 4. Contextual features (time, location, device)
            contextual_features, contextual_metadata = self._extract_contextual_features(transaction_data)
            features['contextual_features'] = contextual_features
            metadata['contextual_features'] = contextual_metadata
            
            # 5. Merchant features
            merchant_features, merchant_metadata = self._extract_merchant_features(transaction_data)
            features['merchant_features'] = merchant_features
            metadata['merchant_features'] = merchant_metadata
            
            # 6. Network/security features
            network_features, network_metadata = self._extract_network_features(transaction_data)
            features['network_features'] = network_features
            metadata['network_features'] = network_metadata
            
            # Combine all features
            combined_features = np.concatenate([
                features['transaction_basic'],
                features['user_behavior'],
                features['velocity_features'],
                features['contextual_features'],
                features['merchant_features'],
                features['network_features']
            ])
            
            # Generate feature names
            feature_names = self._generate_feature_names()
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create feature vector
            feature_vector = FeatureVector(
                features=combined_features,
                feature_names=feature_names,
                transaction_id=transaction_data.get('transaction_id', 'unknown'),
                processing_time_ms=processing_time,
                feature_metadata=metadata if include_metadata else {}
            )
            
            logger.debug(f"Features extracted: {len(combined_features)} features in {processing_time:.2f}ms")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return zero vector as fallback
            return FeatureVector(
                features=np.zeros(self.total_features),
                feature_names=self._generate_feature_names(),
                transaction_id=transaction_data.get('transaction_id', 'unknown'),
                processing_time_ms=0.0,
                feature_metadata={'error': str(e)}
            )
    
    def _extract_basic_features(self, transaction_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Extract basic transaction features."""
        features = []
        metadata = {}
        
        # Amount-based features
        amount = float(transaction_data.get('amount', 0))
        features.extend([
            math.log1p(amount),                    # Log-transformed amount
            amount / 1000.0,                       # Amount in thousands
            min(amount / 10000.0, 1.0),           # Normalized amount (capped at 1)
            1.0 if amount > 5000 else 0.0,        # High amount flag
            1.0 if amount < 10 else 0.0,          # Low amount flag
        ])
        
        # Categorical features (encoded)
        merchant = transaction_data.get('merchant', 'Unknown')
        country = transaction_data.get('country', 'Unknown')  
        device = transaction_data.get('device', 'unknown')
        
        # Simple encoding for categorical features
        features.extend([
            hash(merchant) % 100 / 100.0,         # Merchant hash feature
            hash(country) % 50 / 50.0,            # Country hash feature
            {'mobile': 0.0, 'desktop': 0.5, 'tablet': 0.3}.get(device, 0.8)  # Device encoding
        ])
        
        metadata = {
            'amount': amount,
            'merchant': merchant,
            'country': country,
            'device': device
        }
        
        return np.array(features, dtype=np.float32), metadata
    
    def _extract_user_behavior_features(self, customer_id: str) -> Tuple[np.ndarray, Dict]:
        """Extract user behavior and history features."""
        features = []
        metadata = {}
        
        try:
            # Get user's recent transactions
            recent_transactions = self.transaction_repo.get_user_transactions(
                customer_id, days=30, limit=100
            )
            
            if recent_transactions:
                amounts = [float(t.amount) for t in recent_transactions]
                
                # Statistical features
                features.extend([
                    len(recent_transactions),                               # Transaction count
                    np.mean(amounts),                                      # Average amount
                    np.std(amounts) if len(amounts) > 1 else 0.0,         # Amount std deviation
                    np.max(amounts),                                       # Max amount
                    np.min(amounts),                                       # Min amount
                ])
                
                # Behavioral patterns
                merchants = [t.merchant for t in recent_transactions]
                countries = [t.country_code for t in recent_transactions]
                devices = [t.device_type for t in recent_transactions]
                
                features.extend([
                    len(set(merchants)) / len(recent_transactions),         # Merchant diversity
                    len(set(countries)) / len(recent_transactions),         # Country diversity
                    len(set(devices)) / len(recent_transactions),           # Device diversity
                ])
                
                # Time-based patterns
                hours = [t.transaction_time.hour for t in recent_transactions]
                weekend_count = sum(1 for t in recent_transactions if t.transaction_time.weekday() >= 5)
                
                features.extend([
                    np.std(hours) if len(hours) > 1 else 0.0,             # Time pattern consistency
                    weekend_count / len(recent_transactions),              # Weekend transaction ratio
                ])
                
                metadata = {
                    'transaction_count': len(recent_transactions),
                    'avg_amount': np.mean(amounts),
                    'unique_merchants': len(set(merchants)),
                    'unique_countries': len(set(countries))
                }
            else:
                # New user - default features
                features = [0.0] * self.feature_categories['user_behavior']
                metadata = {'status': 'new_user'}
                
        except Exception as e:
            logger.warning(f"User behavior extraction failed for {customer_id}: {e}")
            features = [0.0] * self.feature_categories['user_behavior']
            metadata = {'error': str(e)}
        
        return np.array(features, dtype=np.float32), metadata
    
    def _extract_velocity_features(self, transaction_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Extract transaction velocity features."""
        features = []
        metadata = {}
        
        customer_id = transaction_data.get('customer_id')
        
        if customer_id:
            try:
                # Get velocity metrics for different time windows
                velocity_1h = self.transaction_repo.get_user_velocity_metrics(
                    customer_id, window_minutes=60
                )
                velocity_6h = self.transaction_repo.get_user_velocity_metrics(
                    customer_id, window_minutes=360
                )
                velocity_24h = self.transaction_repo.get_user_velocity_metrics(
                    customer_id, window_minutes=1440
                )
                
                features.extend([
                    velocity_1h['transaction_count'],                      # Transactions in last hour
                    velocity_1h['total_amount'] / 1000.0,                 # Amount in last hour (in thousands)
                    velocity_6h['transaction_count'],                     # Transactions in last 6 hours
                    velocity_6h['total_amount'] / 1000.0,                # Amount in last 6 hours
                    velocity_24h['transaction_count'],                    # Transactions in last 24 hours
                    velocity_24h['total_amount'] / 1000.0,               # Amount in last 24 hours
                ])
                
                metadata = {
                    'velocity_1h': velocity_1h,
                    'velocity_6h': velocity_6h,
                    'velocity_24h': velocity_24h
                }
                
            except Exception as e:
                logger.warning(f"Velocity extraction failed for {customer_id}: {e}")
                features = [0.0] * self.feature_categories['velocity_features']
                metadata = {'error': str(e)}
        else:
            features = [0.0] * self.feature_categories['velocity_features']
            metadata = {'status': 'no_customer_id'}
        
        return np.array(features, dtype=np.float32), metadata
    
    def _extract_contextual_features(self, transaction_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Extract contextual features (time, location, etc.)."""
        features = []
        metadata = {}
        
        # Time-based features
        timestamp = transaction_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            features.extend([
                hour / 24.0,                                              # Hour of day (normalized)
                day_of_week / 6.0,                                       # Day of week (normalized)
                math.sin(2 * math.pi * hour / 24),                       # Hour cyclical encoding (sin)
                math.cos(2 * math.pi * hour / 24),                       # Hour cyclical encoding (cos)
                1.0 if day_of_week >= 5 else 0.0,                       # Weekend flag
                1.0 if 22 <= hour or hour <= 6 else 0.0,                # Night hours flag
            ])
        else:
            features.extend([0.5, 0.5, 0.0, 1.0, 0.0, 0.0])  # Default values
        
        # Geographic risk (simplified)
        country = transaction_data.get('country', 'Unknown')
        high_risk_countries = {'XX', 'YY', 'ZZ', 'Unknown'}  # These would be real high-risk countries
        
        features.extend([
            1.0 if country in high_risk_countries else 0.0,             # High-risk country flag
            hash(country) % 20 / 20.0,                                  # Country risk score (simplified)
        ])
        
        metadata = {
            'hour': timestamp.hour if timestamp else None,
            'day_of_week': timestamp.weekday() if timestamp else None,
            'country': country,
            'is_high_risk_country': country in high_risk_countries
        }
        
        return np.array(features, dtype=np.float32), metadata
    
    def _extract_merchant_features(self, transaction_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Extract merchant-specific features."""
        features = []
        metadata = {}
        
        merchant = transaction_data.get('merchant', 'Unknown')
        
        # Merchant risk score
        merchant_risk = self.merchant_risk_scores.get(merchant, 0.5)
        
        # Merchant statistics (this would come from historical analysis)
        try:
            merchant_stats = self.transaction_repo.get_merchant_statistics(merchant, days=7)
            
            features.extend([
                merchant_risk,                                            # Merchant risk score
                math.log1p(merchant_stats['transaction_count']),          # Merchant transaction volume
                merchant_stats['avg_amount'] / 1000.0,                   # Merchant average amount
                merchant_stats['unique_customers'] / max(merchant_stats['transaction_count'], 1),  # Customer diversity
                1.0 if merchant_stats['transaction_count'] < 10 else 0.0,  # Low-volume merchant flag
            ])
            
            metadata = {
                'merchant': merchant,
                'merchant_risk': merchant_risk,
                'merchant_stats': merchant_stats
            }
            
        except Exception as e:
            logger.warning(f"Merchant feature extraction failed for {merchant}: {e}")
            features = [merchant_risk, 0.0, 0.0, 0.0, 1.0]  # Default values with high risk for unknown merchants
            metadata = {'merchant': merchant, 'error': str(e)}
        
        return np.array(features, dtype=np.float32), metadata
    
    def _extract_network_features(self, transaction_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Extract network and security features."""
        features = []
        metadata = {}
        
        # IP address features (simplified)
        ip_address = transaction_data.get('ip_address', '127.0.0.1')
        
        # Simple IP hashing for anonymized features
        ip_hash = hash(ip_address) % 1000000
        
        features.extend([
            ip_hash / 1000000.0,                                         # IP address hash feature
            1.0 if ip_address.startswith('127.') else 0.0,              # Local IP flag
            1.0 if ip_address.startswith('10.') or ip_address.startswith('192.168.') else 0.0,  # Private IP flag
        ])
        
        # Session features
        session_id = transaction_data.get('session_id', '')
        features.append(
            1.0 if len(session_id) < 10 else 0.0                        # Short session ID (potentially suspicious)
        )
        
        metadata = {
            'ip_address': ip_address[:8] + '***' if len(ip_address) > 8 else ip_address,  # Masked IP
            'session_id_length': len(session_id)
        }
        
        return np.array(features, dtype=np.float32), metadata
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive names for all features."""
        names = []
        
        # Basic transaction features
        names.extend([
            'amount_log', 'amount_thousands', 'amount_normalized', 
            'high_amount_flag', 'low_amount_flag', 'merchant_hash',
            'country_hash', 'device_encoding'
        ])
        
        # User behavior features
        names.extend([
            'user_transaction_count', 'user_avg_amount', 'user_amount_std',
            'user_max_amount', 'user_min_amount', 'merchant_diversity',
            'country_diversity', 'device_diversity', 'time_pattern_consistency',
            'weekend_transaction_ratio'
        ])
        
        # Velocity features
        names.extend([
            'transactions_1h', 'amount_1h', 'transactions_6h',
            'amount_6h', 'transactions_24h', 'amount_24h'
        ])
        
        # Contextual features
        names.extend([
            'hour_normalized', 'day_of_week_normalized', 'hour_sin',
            'hour_cos', 'weekend_flag', 'night_hours_flag',
            'high_risk_country_flag', 'country_risk_score'
        ])
        
        # Merchant features
        names.extend([
            'merchant_risk_score', 'merchant_volume', 'merchant_avg_amount',
            'merchant_customer_diversity', 'low_volume_merchant_flag'
        ])
        
        # Network features
        names.extend([
            'ip_hash_feature', 'local_ip_flag', 'private_ip_flag',
            'short_session_id_flag'
        ])
        
        return names
    
    def get_feature_importance_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get mapping of features to their categories and descriptions."""
        feature_names = self._generate_feature_names()
        categories = []
        
        # Build category mapping
        for category, count in self.feature_categories.items():
            categories.extend([category] * count)
        
        return {
            name: {
                'category': category,
                'index': i,
                'description': self._get_feature_description(name)
            }
            for i, (name, category) in enumerate(zip(feature_names, categories))
        }
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for a feature."""
        descriptions = {
            'amount_log': 'Log-transformed transaction amount',
            'amount_thousands': 'Transaction amount in thousands',
            'amount_normalized': 'Normalized transaction amount (0-1)',
            'high_amount_flag': 'Binary flag for high-value transactions (>5000)',
            'low_amount_flag': 'Binary flag for low-value transactions (<10)',
            'merchant_hash': 'Hash-based encoding of merchant name',
            'country_hash': 'Hash-based encoding of country code',
            'device_encoding': 'Encoded device type (mobile/desktop/tablet)',
            'user_transaction_count': 'Number of user transactions in last 30 days',
            'user_avg_amount': 'User\'s average transaction amount',
            'merchant_risk_score': 'Historical risk score for merchant',
            'transactions_1h': 'Number of transactions by user in last hour',
            'weekend_flag': 'Binary flag for weekend transactions',
            'night_hours_flag': 'Binary flag for night-time transactions (10pm-6am)',
            'high_risk_country_flag': 'Binary flag for high-risk countries'
        }
        return descriptions.get(feature_name, f'Feature: {feature_name}')
"""
Advanced fraud detection ML pipeline with ensemble methods and feature engineering.
Includes IsolationForest, RandomForest, XGBoost, and sophisticated feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import joblib
import logging

# ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
import optuna
import shap

from ..core.config import get_settings
from ..core.logging import get_logger, performance_logger
from ..models import Transaction, Prediction

logger = get_logger(__name__)
settings = get_settings()


class FeatureEngineering:
    """
    Advanced feature engineering for fraud detection.
    Generates behavioral, velocity, and contextual features.
    """
    
    def __init__(self):
        self.feature_columns = []
        self.scalers = {}
        self.encoders = {}
    
    def extract_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features from transaction data.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Starting feature extraction...")
        start_time = datetime.utcnow()
        
        df = transactions_df.copy()
        
        # Basic transaction features
        df['amount_log'] = np.log1p(df['amount'])
        df['hour'] = pd.to_datetime(df['transaction_time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['transaction_time']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Velocity features (transaction frequency and patterns)
        df = self._add_velocity_features(df)
        
        # Behavioral features (user spending patterns)
        df = self._add_behavioral_features(df)
        
        # Merchant and location features
        df = self._add_merchant_features(df)
        
        # Device and IP features
        df = self._add_device_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Amount-based features
        df = self._add_amount_features(df)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in [
            'id', 'user_id', 'transaction_time', 'created_at', 'updated_at'
        ]]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        performance_logger.log_performance_metric(
            operation="feature_extraction",
            duration_seconds=processing_time,
            record_count=len(df),
            additional_data={"feature_count": len(self.feature_columns)}
        )
        
        logger.info(f"Feature extraction completed. Generated {len(self.feature_columns)} features")
        return df
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction velocity features."""
        df = df.sort_values(['user_id', 'transaction_time'])
        
        # Transactions per user in last N minutes/hours/days
        for window in [10, 30, 60, 360, 1440]:  # 10min, 30min, 1h, 6h, 24h
            df[f'tx_count_last_{window}m'] = (
                df.groupby('user_id')['transaction_time']
                .rolling(f'{window}T', on='transaction_time')
                .count()
                .fillna(0)
                .values
            )
        
        # Amount velocity
        for window in [60, 360, 1440]:  # 1h, 6h, 24h
            df[f'amount_sum_last_{window}m'] = (
                df.groupby('user_id')['amount']
                .rolling(f'{window}T', on='transaction_time')
                .sum()
                .fillna(0)
                .values
            )
        
        # Time between transactions
        df['time_since_last_tx'] = (
            df.groupby('user_id')['transaction_time']
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user behavioral features."""
        
        # User historical statistics
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            'hour': 'mean',
            'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).fillna(0)
        
        user_stats.columns = [f'user_{col[0]}_{col[1]}' for col in user_stats.columns]
        user_stats = user_stats.reset_index()
        
        # Merge user stats
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Deviation from user's normal behavior
        df['amount_zscore'] = (
            (df['amount'] - df['user_amount_mean']) / 
            (df['user_amount_std'] + 1e-8)
        )
        
        df['hour_deviation'] = abs(df['hour'] - df['user_hour_mean'])
        
        return df
    
    def _add_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add merchant and location features."""
        
        # Merchant transaction statistics
        merchant_stats = df.groupby('merchant_name').agg({
            'amount': ['mean', 'std', 'count'],
            'user_id': 'nunique'
        }).fillna(0)
        
        merchant_stats.columns = [f'merchant_{col[0]}_{col[1]}' for col in merchant_stats.columns]
        merchant_stats = merchant_stats.reset_index()
        
        df = df.merge(merchant_stats, on='merchant_name', how='left')
        
        # Country and city features
        df['is_high_risk_country'] = df['country_code'].isin([
            'NG', 'GH', 'PH', 'ID', 'MY'  # Example high-risk countries
        ]).astype(int)
        
        # Location consistency
        user_locations = (
            df.groupby('user_id')['country_code']
            .apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'XX')
            .to_dict()
        )
        
        df['is_different_country'] = (
            df.apply(lambda row: row['country_code'] != user_locations.get(row['user_id'], 'XX'), axis=1)
            .astype(int)
        )
        
        return df
    
    def _add_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add device and IP features."""
        
        # Device consistency
        user_devices = (
            df.groupby('user_id')['device_fingerprint']
            .apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown')
            .to_dict()
        )
        
        df['is_new_device'] = (
            df.apply(lambda row: row['device_fingerprint'] != user_devices.get(row['user_id'], 'unknown'), axis=1)
            .astype(int)
        )
        
        # IP address features
        ip_stats = df.groupby('ip_address').agg({
            'user_id': 'nunique',
            'amount': 'count'
        }).rename(columns={'user_id': 'ip_unique_users', 'amount': 'ip_tx_count'})
        
        df = df.merge(ip_stats, on='ip_address', how='left')
        
        df['is_shared_ip'] = (df['ip_unique_users'] > 1).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced time-based features."""
        
        df['transaction_datetime'] = pd.to_datetime(df['transaction_time'])
        
        # Cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month and day features
        df['month'] = df['transaction_datetime'].dt.month
        df['day_of_month'] = df['transaction_datetime'].dt.day
        
        # Holiday and special day features (simplified)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add amount-based features."""
        
        # Round number detection
        df['is_round_amount'] = (df['amount'] % 1 == 0).astype(int)
        df['is_round_10'] = (df['amount'] % 10 == 0).astype(int)
        df['is_round_100'] = (df['amount'] % 100 == 0).astype(int)
        
        # Amount categories
        amount_percentiles = df['amount'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0] + amount_percentiles.tolist() + [float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme', 'suspicious']
        )
        
        # Encode amount category
        le = LabelEncoder()
        df['amount_category_encoded'] = le.fit_transform(df['amount_category'].astype(str))
        self.encoders['amount_category'] = le
        
        return df
    
    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for ML training with proper encoding and scaling.
        
        Args:
            df: DataFrame with extracted features
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        # Select numeric features
        numeric_features = df[self.feature_columns].select_dtypes(include=[np.number])
        
        # Handle categorical features if any
        categorical_cols = df[self.feature_columns].select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        # Combine all features
        feature_df = pd.concat([
            numeric_features,
            df[categorical_cols]
        ], axis=1)
        
        # Scale features
        scaler = StandardScaler()
        feature_array = scaler.fit_transform(feature_df)
        self.scalers['feature_scaler'] = scaler
        
        return feature_array, feature_df.columns.tolist()


class BaseModel(ABC):
    """Abstract base class for fraud detection models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.training_metrics = {}
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if y_proba.ndim > 1 else self.predict_proba(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'avg_precision': average_precision_score(y, y_proba)
        }


class IsolationForestModel(BaseModel):
    """Isolation Forest for anomaly detection."""
    
    def __init__(self):
        super().__init__("isolation_forest")
        self.contamination = settings.ml.isolation_forest_contamination
    
    def train(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Train Isolation Forest (unsupervised)."""
        logger.info("Training Isolation Forest model...")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=settings.ml.random_seed,
            n_jobs=-1
        )
        
        self.model.fit(X)
        self.is_trained = True
        
        # Calculate anomaly scores
        anomaly_scores = self.model.decision_function(X)
        
        return {
            "model_type": "isolation_forest",
            "contamination": self.contamination,
            "anomaly_score_mean": np.mean(anomaly_scores),
            "anomaly_score_std": np.std(anomaly_scores)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        # Convert to binary (1 for fraud, 0 for normal)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Decision function gives anomaly scores (lower = more anomalous)
        anomaly_scores = self.model.decision_function(X)
        
        # Convert to probabilities (sigmoid transformation)
        probabilities = 1 / (1 + np.exp(anomaly_scores))
        
        return np.column_stack([1 - probabilities, probabilities])


class RandomForestModel(BaseModel):
    """Random Forest classifier for fraud detection."""
    
    def __init__(self):
        super().__init__("random_forest")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Hyperparameter tuning with GridSearch
        if settings.ml.hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            rf = RandomForestClassifier(random_state=settings.ml.random_seed, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, 
                cv=3, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=settings.ml.random_seed,
                n_jobs=-1
            )
            self.model.fit(X, y)
            best_params = {}
        
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1')
        
        return {
            "model_type": "random_forest",
            "best_params": best_params,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "feature_importance_top10": self.get_top_features(10)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most important features."""
        if self.feature_importance is None:
            return []
        
        # This would need feature names passed from the pipeline
        feature_importance_pairs = list(enumerate(self.feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"feature_index": idx, "importance": importance}
            for idx, importance in feature_importance_pairs[:n]
        ]


class XGBoostModel(BaseModel):
    """XGBoost classifier for fraud detection."""
    
    def __init__(self):
        super().__init__("xgboost")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model with Optuna optimization."""
        logger.info("Training XGBoost model...")
        
        if settings.ml.hyperparameter_tuning:
            # Use Optuna for hyperparameter optimization
            study = optuna.create_study(direction='maximize')
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': settings.ml.random_seed
                }
                
                model = xgb.XGBClassifier(**params)
                scores = cross_val_score(model, X, y, cv=3, scoring='f1')
                return scores.mean()
            
            study.optimize(objective, n_trials=20)
            best_params = study.best_params
            
            self.model = xgb.XGBClassifier(**best_params)
        else:
            # Use default parameters
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=settings.ml.random_seed
            )
            best_params = {}
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1')
        
        return {
            "model_type": "xgboost",
            "best_params": best_params,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "feature_importance_top10": self.get_top_features(10)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most important features."""
        if self.feature_importance is None:
            return []
        
        feature_importance_pairs = list(enumerate(self.feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"feature_index": idx, "importance": importance}
            for idx, importance in feature_importance_pairs[:n]
        ]


# Export key components
__all__ = [
    'FeatureEngineering',
    'IsolationForestModel',
    'RandomForestModel', 
    'XGBoostModel',
    'BaseModel'
]
"""
Advanced ML Training Pipeline with Multiple Models and Ensemble Methods
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

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import shap

from app.core.config import get_settings
from ml.features.feature_engineering import AdvancedFeatureExtractor

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    training_time: float
    cv_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    metrics: ModelMetrics
    model_type: str
    model_version: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    training_data_info: Dict[str, Any]


class FraudModelTrainer:
    """Advanced model trainer with multiple algorithms and ensemble methods."""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or settings.ml.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'isolation_forest': {
                'class': IsolationForest,
                'params': {
                    'contamination': [0.05, 0.1, 0.15],
                    'n_estimators': [100, 200, 300],
                    'max_samples': ['auto', 0.5, 0.8],
                    'random_state': [settings.ml.random_state]
                },
                'scoring': 'roc_auc'
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None],
                    'random_state': [settings.ml.random_state]
                },
                'scoring': 'f1'
            },
            'xgboost': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'scale_pos_weight': [1, 3, 5],  # For imbalanced data
                    'random_state': [settings.ml.random_state]
                },
                'scoring': 'f1'
            }
        }
        
        # SHAP explainers for model interpretation
        self.explainers = {}
        
    def generate_training_data(
        self, 
        n_samples: int = 10000, 
        fraud_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic training data for fraud detection.
        
        Args:
            n_samples: Total number of samples to generate
            fraud_ratio: Proportion of fraudulent transactions
            
        Returns:
            X: Feature matrix
            y: Labels (0=legitimate, 1=fraud)
            feature_names: List of feature names
        """
        logger.info(f"Generating {n_samples} training samples with {fraud_ratio:.1%} fraud ratio")
        
        # Generate diverse transaction samples
        merchants = [
            "Amazon", "Walmart", "Apple", "Google", "Target", "Best Buy",
            "Netflix", "Spotify", "Uber", "DoorDash", "Unknown", "Test_Merchant"
        ]
        devices = ["mobile", "desktop", "tablet"]
        countries = ["US", "UK", "CA", "AU", "FR", "DE", "JP", "BR", "IN", "Unknown"]
        
        training_data = []
        labels = []
        
        n_fraud = int(n_samples * fraud_ratio)
        n_legitimate = n_samples - n_fraud
        
        # Generate legitimate transactions
        for _ in range(n_legitimate):
            # Normal transaction patterns
            amount = np.random.lognormal(mean=4.0, sigma=1.0)  # $50-$500 typical
            merchant = np.random.choice(merchants[:-2], p=[0.15, 0.12, 0.12, 0.1, 0.08, 0.08, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05])
            device = np.random.choice(devices, p=[0.5, 0.4, 0.1])
            country = np.random.choice(countries[:-1], p=[0.4, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
            
            transaction = {
                "transaction_id": f"legit_{_}",
                "amount": float(amount),
                "merchant": merchant,
                "device": device,
                "country": country,
                "customer_id": f"user_{np.random.randint(1, 1000)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            training_data.append(transaction)
            labels.append(0)  # Legitimate
        
        # Generate fraudulent transactions
        for _ in range(n_fraud):
            # Fraud patterns: higher amounts, unknown merchants, suspicious timing
            if np.random.random() < 0.4:
                # High-value fraud
                amount = np.random.lognormal(mean=8.0, sigma=1.5)  # $1000-$10000
            else:
                # Regular amount but suspicious patterns
                amount = np.random.lognormal(mean=4.5, sigma=1.2)
            
            # Fraud more likely with unknown merchants or at odd hours
            if np.random.random() < 0.3:
                merchant = "Unknown"
            else:
                merchant = np.random.choice(merchants)
            
            device = np.random.choice(devices)
            country = np.random.choice(countries)
            
            transaction = {
                "transaction_id": f"fraud_{_}",
                "amount": float(amount),
                "merchant": merchant,
                "device": device,
                "country": country,
                "customer_id": f"user_{np.random.randint(1, 1000)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            training_data.append(transaction)
            labels.append(1)  # Fraud
        
        # Extract features using the feature extractor
        # Note: In a real implementation, we'd need a database session
        # For training, we'll use a simplified feature extraction
        features_matrix = []
        
        for transaction in training_data:
            # Simplified feature extraction for training
            features = self._extract_simple_features(transaction)
            features_matrix.append(features)
        
        X = np.array(features_matrix)
        y = np.array(labels)
        
        # Generate simple feature names
        feature_names = [
            'amount_log', 'amount_normalized', 'merchant_hash', 'device_encoded',
            'country_hash', 'amount_high_flag', 'amount_low_flag', 'merchant_risk',
            'hour_normalized', 'weekend_flag'
        ]
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Generated training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)} (0=legit, 1=fraud)")
        
        return X, y, feature_names
    
    def _extract_simple_features(self, transaction: Dict[str, Any]) -> List[float]:
        """Simplified feature extraction for training data generation."""
        import math
        
        amount = transaction['amount']
        merchant = transaction['merchant']
        device = transaction['device']
        country = transaction['country']
        
        # Simple merchant risk mapping
        merchant_risks = {
            'Unknown': 0.8, 'Test_Merchant': 0.6, 'Amazon': 0.1, 'Walmart': 0.15,
            'Apple': 0.12, 'Google': 0.18, 'Target': 0.2, 'Best Buy': 0.25,
            'Netflix': 0.1, 'Spotify': 0.1, 'Uber': 0.3, 'DoorDash': 0.25
        }
        
        # Extract timestamp features
        timestamp = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        features = [
            math.log1p(amount),                              # amount_log
            min(amount / 10000.0, 1.0),                     # amount_normalized
            hash(merchant) % 100 / 100.0,                   # merchant_hash
            {'mobile': 0.0, 'desktop': 0.5, 'tablet': 0.3}.get(device, 0.8),  # device_encoded
            hash(country) % 50 / 50.0,                      # country_hash
            1.0 if amount > 5000 else 0.0,                  # amount_high_flag 
            1.0 if amount < 10 else 0.0,                    # amount_low_flag
            merchant_risks.get(merchant, 0.5),              # merchant_risk
            hour / 24.0,                                    # hour_normalized
            1.0 if is_weekend else 0.0,                    # weekend_flag
        ]
        
        return features
    
    def train_individual_model(
        self, 
        model_type: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        hyperparameter_tuning: bool = True
    ) -> TrainingResult:
        """
        Train an individual model with hyperparameter tuning.
        
        Args:
            model_type: Type of model to train ('isolation_forest', 'random_forest', 'xgboost')
            X_train, X_test, y_train, y_test: Training and test data
            feature_names: Names of features
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            TrainingResult with trained model and metrics
        """
        logger.info(f"Training {model_type} model...")
        start_time = time.time()
        
        config = self.model_configs[model_type]
        
        if hyperparameter_tuning:
            # Hyperparameter tuning with GridSearchCV
            logger.info(f"Performing hyperparameter tuning for {model_type}")
            
            # Create pipeline with scaler for tree-based models
            if model_type in ['random_forest', 'xgboost']:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', config['class']())
                ])
                # Prefix parameter names with 'model__'
                param_grid = {f'model__{k}': v for k, v in config['params'].items()}
            else:
                # Isolation Forest doesn't need scaling
                pipeline = config['class']()
                param_grid = config['params']
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=settings.ml.cv_folds,
                scoring=config['scoring'],
                n_jobs=-1,
                verbose=1
            )
            
            # Handle unsupervised case (Isolation Forest)
            if model_type == 'isolation_forest':
                # For unsupervised learning, we fit on X_train only
                grid_search.fit(X_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                # For supervised learning
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
        
        else:
            # Use default parameters
            if model_type == 'isolation_forest':
                best_model = config['class'](
                    contamination=0.1,
                    random_state=settings.ml.random_state
                )
                best_model.fit(X_train)
            else:
                best_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', config['class'](random_state=settings.ml.random_state))
                ])
                best_model.fit(X_train, y_train)
            
            best_params = {}
        
        training_time = time.time() - start_time
        
        # Make predictions
        if model_type == 'isolation_forest':
            # Isolation Forest returns -1 for outliers, 1 for inliers
            y_pred_raw = best_model.predict(X_test)
            y_pred = np.where(y_pred_raw == -1, 1, 0)  # Convert to 0/1
            
            # Get anomaly scores for probabilistic output
            scores = best_model.decision_function(X_test)
            y_proba = np.zeros((len(y_pred), 2))
            y_proba[:, 1] = (0.5 - scores) * 2  # Convert to probability-like scores
            y_proba[:, 1] = np.clip(y_proba[:, 1], 0, 1)
            y_proba[:, 0] = 1 - y_proba[:, 1]
        else:
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)
            else:
                # Fallback for models without predict_proba
                y_proba = np.zeros((len(y_pred), 2))
                y_proba[np.arange(len(y_pred)), y_pred] = 1.0
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_test, y_pred, y_proba, training_time, best_model, X_train, y_train
        )
        
        # Get feature importance
        feature_importance = self._get_feature_importance(
            best_model, feature_names, model_type
        )
        metrics.feature_importance = feature_importance
        
        # Create model version
        model_version = f"{model_type}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training data info
        training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_fraud': np.sum(y_train),
            'fraud_ratio': np.mean(y_train),
            'training_time': training_time
        }
        
        logger.info(f"{model_type} training completed in {training_time:.2f}s")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}, AUC: {metrics.roc_auc:.4f}")
        
        return TrainingResult(
            model=best_model,
            metrics=metrics,
            model_type=model_type,
            model_version=model_version,
            hyperparameters=best_params,
            feature_names=feature_names,
            training_data_info=training_info
        )
    
    def train_ensemble_model(
        self,
        individual_results: List[TrainingResult],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> TrainingResult:
        """
        Create and train an ensemble model from individual models.
        
        Args:
            individual_results: Results from individual model training
            X_train, X_test, y_train, y_test: Training and test data
            feature_names: Names of features
            
        Returns:
            TrainingResult for ensemble model
        """
        logger.info("Creating ensemble model...")
        start_time = time.time()
        
        # Create voting classifier from individual models
        estimators = []
        weights = []
        
        for i, result in enumerate(individual_results):
            if result.model_type != 'isolation_forest':  # Voting classifier needs supervised models
                estimators.append((result.model_type, result.model))
                # Weight by F1 score
                weights.append(result.metrics.f1_score)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability outputs
            weights=weights
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_test, y_pred, y_proba, training_time, ensemble, X_train, y_train
        )
        
        # Get ensemble feature importance (average of individual importances)
        feature_importance = self._get_ensemble_feature_importance(
            individual_results, feature_names
        )
        metrics.feature_importance = feature_importance
        
        model_version = f"ensemble_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_fraud': np.sum(y_train),
            'fraud_ratio': np.mean(y_train),
            'training_time': training_time,
            'component_models': [r.model_type for r in individual_results],
            'ensemble_weights': weights.tolist()
        }
        
        logger.info(f"Ensemble training completed in {training_time:.2f}s")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}, AUC: {metrics.roc_auc:.4f}")
        
        return TrainingResult(
            model=ensemble,
            metrics=metrics,
            model_type='ensemble',
            model_version=model_version,
            hyperparameters={'weights': weights.tolist(), 'voting': 'soft'},
            feature_names=feature_names,
            training_data_info=training_info
        )
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        training_time: float,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> ModelMetrics:
        """Calculate comprehensive model performance metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        except:
            roc_auc = 0.5  # Random classifier baseline
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        except:
            cv_scores = [f1]  # Fallback for unsupported models
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            training_time=training_time,
            cv_scores=cv_scores.tolist()
        )
    
    def _get_feature_importance(
        self, 
        model: Any, 
        feature_names: List[str], 
        model_type: str
    ) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        
        try:
            if model_type == 'isolation_forest':
                # Isolation Forest doesn't have feature importance
                # Return uniform importance
                importance = np.ones(len(feature_names)) / len(feature_names)
            elif hasattr(model, 'feature_importances_'):
                # Tree-based models (RandomForest, XGBoost)
                if hasattr(model, 'named_steps'):
                    # Pipeline
                    importance = model.named_steps['model'].feature_importances_
                else:
                    importance = model.feature_importances_
            else:
                # Default to uniform importance
                importance = np.ones(len(feature_names)) / len(feature_names)
            
            return dict(zip(feature_names, importance.tolist()))
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            # Return uniform importance as fallback
            uniform_importance = 1.0 / len(feature_names)
            return {name: uniform_importance for name in feature_names}
    
    def _get_ensemble_feature_importance(
        self,
        individual_results: List[TrainingResult],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate ensemble feature importance as weighted average."""
        
        ensemble_importance = {}
        total_weight = 0.0
        
        for result in individual_results:
            if result.metrics.feature_importance:
                weight = result.metrics.f1_score  # Weight by performance
                total_weight += weight
                
                for feature, importance in result.metrics.feature_importance.items():
                    if feature not in ensemble_importance:
                        ensemble_importance[feature] = 0.0
                    ensemble_importance[feature] += importance * weight
        
        # Normalize
        if total_weight > 0:
            for feature in ensemble_importance:
                ensemble_importance[feature] /= total_weight
        
        return ensemble_importance
    
    def save_model(self, result: TrainingResult) -> str:
        """Save trained model to disk."""
        
        model_path = self.models_dir / f"{result.model_version}.joblib"
        metadata_path = self.models_dir / f"{result.model_version}_metadata.json"
        
        # Save model
        joblib.dump(result.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': result.model_type,
            'model_version': result.model_version,
            'feature_names': result.feature_names,
            'hyperparameters': result.hyperparameters,
            'training_data_info': result.training_data_info,
            'metrics': {
                'accuracy': result.metrics.accuracy,
                'precision': result.metrics.precision,
                'recall': result.metrics.recall,
                'f1_score': result.metrics.f1_score,
                'roc_auc': result.metrics.roc_auc,
                'training_time': result.metrics.training_time,
                'cv_scores': result.metrics.cv_scores
            },
            'feature_importance': result.metrics.feature_importance,
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        return str(model_path)
    
    def full_training_pipeline(
        self,
        n_samples: int = 10000,
        fraud_ratio: float = 0.1,
        test_size: float = 0.2,
        hyperparameter_tuning: bool = True,
        save_models: bool = True
    ) -> Dict[str, TrainingResult]:
        """
        Run complete training pipeline with all models.
        
        Args:
            n_samples: Number of training samples to generate
            fraud_ratio: Proportion of fraudulent samples
            test_size: Proportion of data for testing
            hyperparameter_tuning: Whether to tune hyperparameters
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of training results by model type
        """
        logger.info("Starting full training pipeline...")
        
        # Generate training data
        X, y, feature_names = self.generate_training_data(n_samples, fraud_ratio)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=settings.ml.random_state, stratify=y
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train individual models
        results = {}
        individual_results = []
        
        for model_type in ['isolation_forest', 'random_forest', 'xgboost']:
            try:
                result = self.train_individual_model(
                    model_type, X_train, X_test, y_train, y_test, 
                    feature_names, hyperparameter_tuning
                )
                results[model_type] = result
                individual_results.append(result)
                
                if save_models:
                    self.save_model(result)
                    
            except Exception as e:
                logger.error(f"Training failed for {model_type}: {e}")
        
        # Train ensemble model
        if len(individual_results) >= 2:
            try:
                ensemble_result = self.train_ensemble_model(
                    individual_results, X_train, X_test, y_train, y_test, feature_names
                )
                results['ensemble'] = ensemble_result
                
                if save_models:
                    self.save_model(ensemble_result)
                    
            except Exception as e:
                logger.error(f"Ensemble training failed: {e}")
        
        # Log summary
        logger.info("Training pipeline completed!")
        for model_type, result in results.items():
            logger.info(
                f"{model_type}: Accuracy={result.metrics.accuracy:.4f}, "
                f"F1={result.metrics.f1_score:.4f}, AUC={result.metrics.roc_auc:.4f}"
            )
        
        return results
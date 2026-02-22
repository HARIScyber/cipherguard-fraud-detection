"""
ML pipeline orchestrator with ensemble methods and model management.
Provides training, inference, and model lifecycle management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path
import logging

# ML libraries
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap

from .models import (
    FeatureEngineering, IsolationForestModel, 
    RandomForestModel, XGBoostModel, BaseModel
)
from ..core.config import get_settings
from ..core.logging import get_logger, performance_logger
from ..models import RiskLevel

logger = get_logger(__name__)
settings = get_settings()


class EnsembleModel:
    """
    Ensemble model combining multiple fraud detection algorithms.
    Uses weighted voting and sophisticated model combination.
    """
    
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForestModel(),
            'random_forest': RandomForestModel(),
            'xgboost': XGBoostModel()
        }
        
        self.model_weights = {
            'isolation_forest': 0.2,  # Lower weight for unsupervised
            'random_forest': 0.4,     # Good baseline performance
            'xgboost': 0.4           # Often best performance
        }
        
        self.is_trained = False
        self.ensemble_metrics = {}
        self.feature_names = []
        self.shap_explainer = None
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ensemble models...")
        start_time = datetime.utcnow()
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        training_results = {}
        
        # Train individual models
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                if model_name == 'isolation_forest':
                    # Isolation Forest is unsupervised
                    result = model.train(X)
                else:
                    result = model.train(X, y)
                
                training_results[model_name] = result
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                training_results[model_name] = {"error": str(e)}
        
        # Create SHAP explainer for the best performing supervised model
        try:
            if self.models['xgboost'].is_trained:
                self.shap_explainer = shap.TreeExplainer(self.models['xgboost'].model)
            elif self.models['random_forest'].is_trained:
                self.shap_explainer = shap.TreeExplainer(self.models['random_forest'].model)
        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer: {str(e)}")
        
        self.is_trained = True
        
        # Evaluate ensemble performance
        ensemble_metrics = self.evaluate_ensemble(X, y)
        self.ensemble_metrics = ensemble_metrics
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        performance_logger.log_performance_metric(
            operation="ensemble_training",
            duration_seconds=training_time,
            record_count=len(X),
            additional_data={
                "models_trained": len([m for m in self.models.values() if m.is_trained]),
                "feature_count": X.shape[1]
            }
        )
        
        return {
            "individual_models": training_results,
            "ensemble_metrics": ensemble_metrics,
            "training_time_seconds": training_time,
            "models_trained": len([m for m in self.models.values() if m.is_trained])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0 = legitimate, 1 = fraud)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = {}
        
        # Get predictions from each trained model
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No trained models available for prediction")
        
        # Weighted voting
        ensemble_pred = np.zeros(X.shape[0])
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        # Normalize and threshold
        ensemble_pred /= total_weight
        return (ensemble_pred >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities using ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix with shape (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        probabilities = {}
        
        # Get probabilities from each trained model
        for model_name, model in self.models.items():
            if model.is_trained:
                try:
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        # For binary classification, convert to 2D
                        probabilities[model_name] = np.column_stack([1 - proba, proba])
                    else:
                        probabilities[model_name] = proba
                except Exception as e:
                    logger.error(f"Probability prediction failed for {model_name}: {str(e)}")
        
        if not probabilities:
            raise ValueError("No trained models available for probability prediction")
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros((X.shape[0], 2))
        total_weight = 0
        
        for model_name, proba in probabilities.items():
            weight = self.model_weights.get(model_name, 1.0)
            ensemble_proba += weight * proba
            total_weight += weight
        
        ensemble_proba /= total_weight
        return ensemble_proba
    
    def get_explanation(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """
        Get SHAP explanations for predictions.
        
        Args:
            X: Feature matrix
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation dictionary
        """
        if not self.shap_explainer:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Limit samples for performance
            X_explain = X[:max_samples] if len(X) > max_samples else X
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # Handle multi-class output (take positive class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class (fraud)
            
            # Calculate feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Get top features
            top_features_idx = np.argsort(feature_importance)[::-1][:10]
            top_features = [
                {
                    "feature": self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}",
                    "importance": float(feature_importance[idx])
                }
                for idx in top_features_idx
            ]
            
            return {
                "shap_values": shap_values.tolist(),
                "feature_importance": feature_importance.tolist(),
                "top_features": top_features,
                "feature_names": self.feature_names
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        try:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)[:, 1]
            
            # Basic metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, average_precision_score
            )
            
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred)),
                'f1_score': float(f1_score(y, y_pred)),
                'roc_auc': float(roc_auc_score(y, y_proba)),
                'avg_precision': float(average_precision_score(y, y_proba))
            }
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {str(e)}")
            return {"error": str(e)}


class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline with training and inference capabilities.
    """
    
    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.ensemble_model = EnsembleModel()
        self.is_trained = False
        self.model_version = None
        self.thresholds = {
            'fraud': settings.ml.fraud_threshold,
            'high_risk': settings.ml.high_risk_threshold,
            'medium_risk': settings.ml.medium_risk_threshold
        }
    
    def train_pipeline(
        self, 
        transactions_df: pd.DataFrame, 
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the complete fraud detection pipeline.
        
        Args:
            transactions_df: DataFrame with transaction data
            labels: Optional labels for supervised learning
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting fraud detection pipeline training...")
        pipeline_start = datetime.utcnow()
        
        try:
            # Feature engineering
            logger.info("Extracting features...")
            features_df = self.feature_engineering.extract_transaction_features(transactions_df)
            
            # Prepare features for training
            X, feature_names = self.feature_engineering.prepare_features_for_training(features_df)
            
            # Generate labels if not provided (for semi-supervised learning)
            if labels is None:
                logger.info("No labels provided, using anomaly detection for initial labeling...")
                # Use simple statistical outlier detection as initial labels
                amount_threshold = np.percentile(transactions_df['amount'], 99)
                velocity_threshold = 10  # Simplified
                
                labels = (
                    (transactions_df['amount'] > amount_threshold) |
                    (transactions_df.groupby('user_id').cumcount() > velocity_threshold)
                ).astype(int).values
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, labels, 
                test_size=0.2, 
                random_state=settings.ml.random_seed,
                stratify=labels
            )
            
            logger.info(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
            logger.info(f"Fraud rate in training: {y_train.mean():.2%}")
            
            # Train ensemble model
            training_results = self.ensemble_model.train(X_train, y_train, feature_names)
            
            # Validate on holdout set
            validation_metrics = self.ensemble_model.evaluate_ensemble(X_val, y_val)
            
            # Set pipeline as trained
            self.is_trained = True
            self.model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Calculate total training time
            total_time = (datetime.utcnow() - pipeline_start).total_seconds()
            
            results = {
                "pipeline_version": self.model_version,
                "feature_count": len(feature_names),
                "training_samples": X_train.shape[0],
                "validation_samples": X_val.shape[0],
                "fraud_rate": float(y_train.mean()),
                "training_results": training_results,
                "validation_metrics": validation_metrics,
                "total_training_time_seconds": total_time,
                "thresholds": self.thresholds
            }
            
            logger.info(f"Pipeline training completed successfully in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline training failed: {str(e)}", exc_info=True)
            raise
    
    def predict_fraud(
        self, 
        transaction_data: Dict[str, Any],
        return_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction_data: Dictionary with transaction features
            return_explanation: Whether to include SHAP explanations
            
        Returns:
            Prediction results dictionary
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")
        
        start_time = datetime.utcnow()
        
        try:
            # Convert to DataFrame for feature engineering
            df = pd.DataFrame([transaction_data])
            
            # Extract features
            features_df = self.feature_engineering.extract_transaction_features(df)
            X, _ = self.feature_engineering.prepare_features_for_training(features_df)
            
            # Get predictions and probabilities
            fraud_probability = float(self.ensemble_model.predict_proba(X)[0, 1])
            is_fraud = fraud_probability >= self.thresholds['fraud']
            
            # Determine risk level
            if fraud_probability >= self.thresholds['fraud']:
                risk_level = RiskLevel.CRITICAL
            elif fraud_probability >= self.thresholds['high_risk']:
                risk_level = RiskLevel.HIGH
            elif fraud_probability >= self.thresholds['medium_risk']:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Get feature importance and explanations
            explanation_data = {}
            if return_explanation:
                explanation_data = self.ensemble_model.get_explanation(X, max_samples=1)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # ms
            
            result = {
                "fraud_probability": fraud_probability,
                "is_fraud": is_fraud,
                "risk_level": risk_level.value,
                "confidence_score": min(max(abs(fraud_probability - 0.5) * 2, 0.5), 1.0),  # Confidence based on distance from threshold
                "processing_time_ms": processing_time,
                "model_version": self.model_version,
                "thresholds_used": self.thresholds.copy()
            }
            
            # Add explanations if requested
            if return_explanation and not explanation_data.get("error"):
                result.update({
                    "feature_importance": explanation_data.get("feature_importance", []),
                    "top_risk_factors": explanation_data.get("top_features", [])[:5],  # Top 5 risk factors
                    "shap_values": explanation_data.get("shap_values", [])
                })
            
            # Log performance metric
            performance_logger.log_performance_metric(
                operation="fraud_prediction",
                duration_seconds=processing_time / 1000,
                record_count=1,
                additional_data={
                    "fraud_probability": fraud_probability,
                    "risk_level": risk_level.value
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fraud prediction failed: {str(e)}", exc_info=True)
            raise
    
    def save_pipeline(self, model_path: str) -> None:
        """Save trained pipeline to disk."""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before saving")
        
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save ensemble model
            ensemble_path = model_dir / "ensemble_model.joblib"
            joblib.dump(self.ensemble_model, ensemble_path)
            
            # Save feature engineering components
            feature_eng_path = model_dir / "feature_engineering.joblib"
            joblib.dump(self.feature_engineering, feature_eng_path)
            
            # Save metadata
            metadata = {
                "model_version": self.model_version,
                "thresholds": self.thresholds,
                "is_trained": self.is_trained,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            metadata_path = model_dir / "metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Pipeline saved successfully to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline: {str(e)}")
            raise
    
    def load_pipeline(self, model_path: str) -> None:
        """Load trained pipeline from disk."""
        try:
            model_dir = Path(model_path)
            
            # Load ensemble model
            ensemble_path = model_dir / "ensemble_model.joblib"
            self.ensemble_model = joblib.load(ensemble_path)
            
            # Load feature engineering components
            feature_eng_path = model_dir / "feature_engineering.joblib"
            self.feature_engineering = joblib.load(feature_eng_path)
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_version = metadata.get("model_version")
            self.thresholds = metadata.get("thresholds", self.thresholds)
            self.is_trained = metadata.get("is_trained", False)
            
            logger.info(f"Pipeline loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {str(e)}")
            raise


# Global pipeline instance
fraud_detection_pipeline = FraudDetectionPipeline()

# Export key components
__all__ = [
    'EnsembleModel',
    'FraudDetectionPipeline',
    'fraud_detection_pipeline'
]
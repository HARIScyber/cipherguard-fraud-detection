"""
Advanced ML Models Module - Phase 5: Next-Level Fraud Detection
Deep learning models and ensemble methods for enhanced fraud detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

# Global instance for easy access
_advanced_detector = None

logger = logging.getLogger(__name__)

class AdvancedFraudDetector:
    """
    Advanced fraud detection using deep learning and ensemble methods.
    Combines neural networks, XGBoost, and traditional ML for superior performance.
    """

    def __init__(self, input_dim: int = 6, models_dir: str = "models"):
        self.input_dim = input_dim
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # Initialize models
        self.neural_network = None
        self.xgboost_model = None
        self.ensemble_model = None

        # Model performance tracking
        self.model_metrics = {}

    def build_neural_network(self) -> keras.Model:
        """
        Build a deep neural network for fraud detection.
        Uses autoencoders and dense layers for anomaly detection.
        """
        logger.info("Building neural network model...")

        # Input layer
        inputs = keras.Input(shape=(self.input_dim,))

        # Encoder layers
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(16, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Bottleneck
        encoded = layers.Dense(8, activation='relu')(x)

        # Decoder layers (for autoencoder reconstruction)
        x = layers.Dense(16, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Output layers
        reconstruction = layers.Dense(self.input_dim, activation='sigmoid')(x)
        fraud_score = layers.Dense(1, activation='sigmoid', name='fraud_score')(encoded)

        # Create model with multiple outputs
        model = keras.Model(inputs=inputs, outputs=[reconstruction, fraud_score])

        # Compile model
        model.compile(
            optimizer='adam',
            loss={
                'dense_6': 'mse',  # reconstruction loss
                'fraud_score': 'binary_crossentropy'
            },
            loss_weights={
                'dense_6': 0.3,  # weight reconstruction loss less
                'fraud_score': 0.7
            },
            metrics={
                'fraud_score': ['accuracy', tf.keras.metrics.AUC()]
            }
        )

        self.neural_network = model
        logger.info("Neural network built successfully")
        return model

    def build_xgboost_model(self) -> xgb.XGBClassifier:
        """
        Build XGBoost model with optimized hyperparameters.
        """
        logger.info("Building XGBoost model...")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False
        )

        self.xgboost_model = model
        logger.info("XGBoost model built successfully")
        return model

    def build_ensemble_model(self) -> RandomForestClassifier:
        """
        Build ensemble model combining multiple algorithms.
        """
        logger.info("Building ensemble model...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        self.ensemble_model = model
        logger.info("Ensemble model built successfully")
        return model

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train all models and return performance metrics.
        """
        logger.info("Training advanced models...")
        results = {}

        # Train Neural Network
        if self.neural_network is None:
            self.build_neural_network()

        # Prepare targets for neural network (autoencoder + classification)
        y_nn = [X_train, y_train]  # reconstruction target + fraud target
        y_val_nn = [X_val, y_val] if X_val is not None else None

        history = self.neural_network.fit(
            X_train, y_nn,
            validation_data=(X_val, y_val_nn) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        results['neural_network'] = {'history': history.history}

        # Train XGBoost
        if self.xgboost_model is None:
            self.build_xgboost_model()

        self.xgboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None, verbose=False)
        results['xgboost'] = {'trained': True}

        # Train Ensemble
        if self.ensemble_model is None:
            self.build_ensemble_model()

        self.ensemble_model.fit(X_train, y_train)
        results['ensemble'] = {'trained': True}

        # Evaluate models
        if X_val is not None:
            results['evaluation'] = self.evaluate_models(X_val, y_val)

        logger.info("All models trained successfully")
        return results

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data.
        """
        results = {}

        # Neural Network evaluation
        if self.neural_network:
            nn_pred = self.neural_network.predict(X_test)[1].flatten()  # fraud_score output
            nn_auc = roc_auc_score(y_test, nn_pred)
            results['neural_network'] = {
                'auc': nn_auc,
                'predictions': nn_pred
            }

        # XGBoost evaluation
        if self.xgboost_model:
            xgb_pred = self.xgboost_model.predict_proba(X_test)[:, 1]
            xgb_auc = roc_auc_score(y_test, xgb_pred)
            results['xgboost'] = {
                'auc': xgb_auc,
                'predictions': xgb_pred
            }

        # Ensemble evaluation
        if self.ensemble_model:
            ens_pred = self.ensemble_model.predict_proba(X_test)[:, 1]
            ens_auc = roc_auc_score(y_test, ens_pred)
            results['ensemble'] = {
                'auc': ens_auc,
                'predictions': ens_pred
            }

        return results

    def predict_fraud(self, transaction_vector: np.ndarray) -> Dict[str, Any]:
        """
        Predict fraud probability using ensemble of all models.
        """
        if transaction_vector.ndim == 1:
            transaction_vector = transaction_vector.reshape(1, -1)

        predictions = {}

        # Neural Network prediction
        if self.neural_network:
            nn_pred = self.neural_network.predict(transaction_vector, verbose=0)[1].flatten()[0]
            predictions['neural_network'] = float(nn_pred)

        # XGBoost prediction
        if self.xgboost_model:
            xgb_pred = self.xgboost_model.predict_proba(transaction_vector)[0, 1]
            predictions['xgboost'] = float(xgb_pred)

        # Ensemble prediction
        if self.ensemble_model:
            ens_pred = self.ensemble_model.predict_proba(transaction_vector)[0, 1]
            predictions['ensemble'] = float(ens_pred)

        # Ensemble score (average of all models)
        if predictions:
            ensemble_score = np.mean(list(predictions.values()))
            predictions['ensemble_score'] = float(ensemble_score)

            # Determine if fraudulent based on threshold
            is_fraudulent = ensemble_score > 0.5
            predictions['is_fraudulent'] = bool(is_fraudulent)

            # Confidence level
            confidence = abs(ensemble_score - 0.5) * 2  # Scale to [0, 1]
            predictions['confidence'] = float(confidence)

        return predictions

    def save_models(self):
        """Save all trained models to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.neural_network:
            nn_path = os.path.join(self.models_dir, f"neural_network_{timestamp}.h5")
            self.neural_network.save(nn_path)
            logger.info(f"Neural network saved to {nn_path}")

        if self.xgboost_model:
            xgb_path = os.path.join(self.models_dir, f"xgboost_{timestamp}.json")
            self.xgboost_model.save_model(xgb_path)
            logger.info(f"XGBoost model saved to {xgb_path}")

        if self.ensemble_model:
            ens_path = os.path.join(self.models_dir, f"ensemble_{timestamp}.joblib")
            joblib.dump(self.ensemble_model, ens_path)
            logger.info(f"Ensemble model saved to {ens_path}")

    def load_models(self, timestamp: str = None):
        """Load saved models from disk."""
        if timestamp is None:
            # Load latest models
            model_files = os.listdir(self.models_dir)
            timestamps = [f.split('_')[-1].split('.')[0] for f in model_files if '_' in f]
            if timestamps:
                timestamp = max(timestamps)

        if timestamp:
            try:
                # Load neural network
                nn_files = [f for f in os.listdir(self.models_dir) if f.startswith('neural_network') and timestamp in f]
                if nn_files:
                    nn_path = os.path.join(self.models_dir, nn_files[0])
                    self.neural_network = keras.models.load_model(nn_path)
                    logger.info(f"Neural network loaded from {nn_path}")

                # Load XGBoost
                xgb_files = [f for f in os.listdir(self.models_dir) if f.startswith('xgboost') and timestamp in f]
                if xgb_files:
                    xgb_path = os.path.join(self.models_dir, xgb_files[0])
                    self.xgboost_model = xgb.XGBClassifier()
                    self.xgboost_model.load_model(xgb_path)
                    logger.info(f"XGBoost model loaded from {xgb_path}")

                # Load Ensemble
                ens_files = [f for f in os.listdir(self.models_dir) if f.startswith('ensemble') and timestamp in f]
                if ens_files:
                    ens_path = os.path.join(self.models_dir, ens_files[0])
                    self.ensemble_model = joblib.load(ens_path)
                    logger.info(f"Ensemble model loaded from {ens_path}")

            except Exception as e:
                logger.error(f"Error loading models: {e}")

    def generate_synthetic_data(self, n_samples: int = 10000,
                              fraud_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic transaction data for training.
        """
        np.random.seed(42)

        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud

        # Generate normal transactions
        normal_data = np.random.rand(n_normal, self.input_dim)
        normal_labels = np.zeros(n_normal)

        # Generate fraudulent transactions (with different patterns)
        fraud_data = np.random.rand(n_fraud, self.input_dim)
        # Make fraud patterns more extreme
        fraud_data[:, 0] = np.random.beta(2, 5, n_fraud)  # Higher amounts
        fraud_data[:, -1] = np.random.beta(5, 2, n_fraud)  # Higher risk flags
        fraud_labels = np.ones(n_fraud)

        # Combine and shuffle
        X = np.vstack([normal_data, fraud_data])
        y = np.hstack([normal_labels, fraud_labels])

        # Shuffle
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        return X, y


def get_advanced_detector() -> AdvancedFraudDetector:
    """Get or create the global advanced fraud detector instance."""
    return AdvancedFraudDetector()</content>
<parameter name="filePath">d:\cipherguard-fraud-poc\app\advanced_models.py
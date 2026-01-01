"""
Fraud Detection Service - Advanced Ensemble Fraud Detection
Microservice for running advanced fraud detection with ensemble ML models and Kafka streaming
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import xgboost as xgb
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    print("Warning: TensorFlow not available, neural network model disabled")
import os
import asyncio
import threading
from kafka import KafkaConsumer, KafkaProducer
from cryptography.fernet import Fernet
import base64
import json
import joblib
import pandas as pd

# Phase 4: Production Deployment imports
try:
    from monitoring import get_metrics_collector, track_model_prediction, track_fraud_detection
    metrics_collector = get_metrics_collector()
except ImportError:
    # Fallback if monitoring module not available
    metrics_collector = None
    track_model_prediction = lambda *args: lambda func: func
    track_fraud_detection = lambda *args: lambda func: func

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CipherGuard Advanced Fraud Detection Service",
    description="Ensemble ML fraud detection with real-time updates and A/B testing",
    version="0.2.0"
)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
EMBEDDING_TOPIC = 'transaction.embedded'
FRAUD_TOPIC = 'transaction.fraud'
MODEL_UPDATE_TOPIC = 'model.updates'

# Encryption setup (Phase 2: Secure Data Pipeline)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'your-32-char-encryption-key-here!')
cipher = Fernet(base64.urlsafe_b64encode(ENCRYPTION_KEY.encode()[:32].ljust(32, b'\0')))

# Global components
models = {}  # Dictionary to store multiple models for A/B testing
active_model_version = "v1.0"  # Current active model version
consumer = None
producer = None

def encrypt_data(data: str) -> str:
    """Encrypt data using Fernet symmetric encryption."""
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data using Fernet symmetric encryption."""
    return cipher.decrypt(encrypted_data.encode()).decode()

def get_kafka_producer():
    """Get or create Kafka producer singleton."""
    global producer
    if producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            logger.info(f"Connected to Kafka producer at {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka producer: {e}")
            producer = None
    return producer

def get_kafka_consumer():
    """Get or create Kafka consumer singleton."""
    global consumer
    if consumer is None:
        try:
            consumer = KafkaConsumer(
                EMBEDDING_TOPIC,
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                group_id='fraud-detection-service',
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',
                enable_auto_commit=True
            )
            logger.info(f"Connected to Kafka consumer at {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka consumer: {e}")
            consumer = None
    return consumer

def initialize_models():
    """Initialize ensemble of ML models for fraud detection."""
    global models

    logger.info("Initializing ensemble ML models...")

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 5000

    # Create realistic transaction features
    data = []
    labels = []

    for _ in range(n_samples):
        # Normal transaction features
        amount = np.random.exponential(100)  # Most transactions small
        time_of_day = np.random.uniform(0, 1)
        merchant_hash = np.random.uniform(0, 1)
        device_hash = np.random.uniform(0, 1)
        country_hash = np.random.uniform(0, 1)
        velocity_1h = np.random.poisson(2)  # Transactions per hour
        velocity_24h = np.random.poisson(10)  # Transactions per day
        amount_std = np.random.uniform(0, 50)  # Amount standard deviation

        # Create feature vector
        features = np.array([
            amount, time_of_day, merchant_hash, device_hash, country_hash,
            velocity_1h, velocity_24h, amount_std
        ])

        # L2 normalize
        features = features / np.linalg.norm(features)

        # Label as fraud based on suspicious patterns
        is_fraud = (
            amount > 500 or  # High amount
            velocity_1h > 5 or  # High velocity
            (amount > 200 and velocity_24h > 15) or  # High amount + high daily velocity
            np.random.random() < 0.05  # 5% base fraud rate
        )

        data.append(features)
        labels.append(1 if is_fraud else 0)

    X = np.array(data)
    y = np.array(labels)

    logger.info(f"Generated {len(X)} training samples with {sum(y)} fraud cases")

    # Model 1: Isolation Forest (Unsupervised)
    isolation_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    isolation_forest.fit(X)
    models['isolation_forest'] = isolation_forest

    # Model 2: Random Forest (Supervised)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X, y)
    models['random_forest'] = rf_model

    # Model 3: XGBoost (Gradient Boosting)
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=len(y)/sum(y),  # Handle class imbalance
        random_state=42
    )
    xgb_model.fit(X, y)
    models['xgboost'] = xgb_model

    # Model 4: Neural Network (Deep Learning) - only if TensorFlow available
    if TENSORFLOW_AVAILABLE and keras is not None:
        nn_model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )

        nn_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        models['neural_network'] = nn_model
    else:
        logger.warning("Neural network model skipped - TensorFlow not available")

    logger.info(f"Initialized {len(models)} ensemble models: {list(models.keys())}")

def get_ensemble_prediction(vector: np.ndarray, model_version: str = None) -> Dict:
    """Get predictions from ensemble of models."""
    if not models:
        initialize_models()

    if model_version and model_version in models:
        # Use specific model version for A/B testing
        model = models[model_version]
        if model_version == 'isolation_forest':
            anomaly_score = model.decision_function([vector])[0]
            fraud_prob = 1.0 / (1.0 + np.exp(-anomaly_score))  # Convert to probability
            return {
                'fraud_probability': float(fraud_prob),
                'model_used': model_version,
                'confidence': float(abs(anomaly_score))
            }
        elif model_version == 'neural_network':
            pred = model.predict(vector.reshape(1, -1))[0][0]
            return {
                'fraud_probability': float(pred),
                'model_used': model_version,
                'confidence': float(pred)
            }
        else:
            # Tree-based models
            pred_proba = model.predict_proba([vector])[0][1]
            return {
                'fraud_probability': float(pred_proba),
                'model_used': model_version,
                'confidence': float(pred_proba)
            }

    # Ensemble prediction (weighted average)
    predictions = []
    # Dynamic weights based on available models
    base_weights = {
        'isolation_forest': 0.25,
        'random_forest': 0.375,
        'xgboost': 0.375,
        'neural_network': 0.25
    }
    # Adjust weights if neural network is not available
    available_models = list(models.keys())
    if 'neural_network' not in available_models:
        # Redistribute neural network weight to other models
        nn_weight = base_weights['neural_network']
        remaining_weight = 1.0 - nn_weight
        weights = {}
        for model in available_models:
            weights[model] = base_weights[model] / remaining_weight
    else:
        weights = base_weights

    for model_name, model in models.items():
        try:
            if model_name == 'isolation_forest':
                anomaly_score = model.decision_function([vector])[0]
                fraud_prob = 1.0 / (1.0 + np.exp(-anomaly_score))
            elif model_name == 'neural_network':
                fraud_prob = model.predict(vector.reshape(1, -1))[0][0]
            else:
                fraud_prob = model.predict_proba([vector])[0][1]

            predictions.append({
                'model': model_name,
                'probability': float(fraud_prob),
                'weight': weights.get(model_name, 1.0)
            })
        except Exception as e:
            logger.warning(f"Error getting prediction from {model_name}: {e}")
            continue

    if not predictions:
        return {
            'fraud_probability': 0.5,
            'model_used': 'ensemble',
            'confidence': 0.0,
            'error': 'No models available'
        }

    # Weighted ensemble prediction
    total_weight = sum(p['weight'] for p in predictions)
    ensemble_prob = sum(p['probability'] * p['weight'] for p in predictions) / total_weight

    # Calculate confidence as agreement between models
    probs = [p['probability'] for p in predictions]
    confidence = 1.0 - np.std(probs)  # Lower variance = higher confidence

    return {
        'fraud_probability': float(ensemble_prob),
        'model_used': 'ensemble',
        'confidence': float(confidence),
        'model_predictions': predictions
    }

@track_fraud_detection("ensemble")
def detect_fraud(transaction_id: str, vector: np.ndarray, metadata: Dict = None, model_version: str = None) -> Dict:
    """Run advanced fraud detection on a transaction vector using ensemble models."""
    try:
        # Get ensemble prediction
        ensemble_result = get_ensemble_prediction(vector, model_version)
        fraud_probability = ensemble_result['fraud_probability']
        confidence = ensemble_result['confidence']

        # Mock kNN search (would use CyborgDB in production)
        knn_results = [
            ("txn_001", 0.1),
            ("txn_002", 0.15),
            ("txn_003", 0.2)
        ]

        # Enhanced fraud scoring with behavioral features
        fraud_score = compute_enhanced_fraud_score(vector, knn_results, fraud_probability, confidence)

        # Determine risk level with more granularity
        risk_level = get_risk_level(fraud_score)

        # Additional insights
        insights = generate_fraud_insights(vector, ensemble_result, knn_results)

        result = {
            "transaction_id": transaction_id,
            "is_fraud": fraud_score >= 0.7,  # Higher threshold for fraud detection
            "fraud_score": float(fraud_score),
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "confidence": confidence,
            "model_used": ensemble_result['model_used'],
            "similar_transactions": [tx_id for tx_id, _ in knn_results],
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }

        return result

    except Exception as e:
        logger.error(f"Fraud detection error for {transaction_id}: {e}")
        return {
            "transaction_id": transaction_id,
            "is_fraud": False,
            "fraud_score": 0.0,
            "risk_level": "UNKNOWN",
            "fraud_probability": 0.0,
            "confidence": 0.0,
            "model_used": "error",
            "similar_transactions": [],
            "insights": [],
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def compute_enhanced_fraud_score(vector: np.ndarray, knn_results: List, fraud_probability: float, confidence: float) -> float:
    """Compute enhanced fraud score using multiple signals."""
    # Base fraud probability from ensemble models
    base_score = fraud_probability

    # kNN similarity score
    if knn_results:
        avg_distance = np.mean([dist for _, dist in knn_results])
        knn_score = min(avg_distance * 2, 1.0)  # Normalize distance to [0,1]
    else:
        knn_score = 0.5

    # Behavioral features from vector
    amount = vector[0] if len(vector) > 0 else 0.5
    velocity_1h = vector[5] if len(vector) > 5 else 0.1
    velocity_24h = vector[6] if len(vector) > 6 else 0.5

    # Behavioral risk factors
    behavioral_risk = 0.0
    if amount > 0.8: behavioral_risk += 0.3  # High amount
    if velocity_1h > 0.5: behavioral_risk += 0.2  # High hourly velocity
    if velocity_24h > 0.8: behavioral_risk += 0.2  # High daily velocity

    # Confidence adjustment
    confidence_multiplier = 0.8 + (confidence * 0.4)  # [0.8, 1.2]

    # Weighted ensemble score
    fraud_score = (
        base_score * 0.5 +           # Ensemble model prediction (50%)
        knn_score * 0.2 +            # Similarity to known fraud (20%)
        behavioral_risk * 0.3        # Behavioral risk factors (30%)
    ) * confidence_multiplier

    return min(max(fraud_score, 0.0), 1.0)  # Clamp to [0,1]

def generate_fraud_insights(vector: np.ndarray, ensemble_result: Dict, knn_results: List) -> List[str]:
    """Generate human-readable insights about the fraud detection."""
    insights = []

    # Model agreement insights
    if 'model_predictions' in ensemble_result:
        predictions = ensemble_result['model_predictions']
        if len(predictions) > 1:
            probs = [p['probability'] for p in predictions]
            agreement = 1.0 - np.std(probs)
            if agreement > 0.8:
                insights.append("High model agreement - strong signal")
            elif agreement < 0.5:
                insights.append("Model disagreement - review manually")

    # Behavioral insights
    amount = vector[0] if len(vector) > 0 else 0.5
    velocity_1h = vector[5] if len(vector) > 5 else 0.1
    velocity_24h = vector[6] if len(vector) > 6 else 0.5

    if amount > 0.8:
        insights.append("Unusually high transaction amount")
    if velocity_1h > 0.5:
        insights.append("High transaction velocity (1h)")
    if velocity_24h > 0.8:
        insights.append("High transaction velocity (24h)")

    # Similarity insights
    if knn_results:
        close_matches = [tx_id for tx_id, dist in knn_results if dist < 0.2]
        if close_matches:
            insights.append(f"Similar to {len(close_matches)} known transactions")

    return insights

def get_risk_level(score: float) -> str:
    """Convert fraud score to risk level."""
    if score >= 0.9:
        return "CRITICAL"
    elif score >= 0.75:
        return "HIGH"
    elif score >= 0.6:
        return "MEDIUM"
    elif score >= 0.4:
        return "LOW"
    else:
        return "VERY_LOW"

def kafka_consumer_loop():
    """Background loop to consume embeddings from Kafka."""
    logger.info("Starting Kafka consumer loop...")
    consumer = get_kafka_consumer()
    if not consumer:
        logger.error("Kafka consumer not available, exiting loop")
        return

    try:
        for message in consumer:
            embedding_data = message.value
            logger.info(f"Received embedding from Kafka: {embedding_data['transaction_id']}")
            process_embedding_from_kafka(embedding_data)
    except Exception as e:
        logger.error(f"Kafka consumer loop error: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("=== CipherGuard Fraud Detection Service Starting ===")

    # Initialize ML model
    initialize_model()

    # Start Kafka consumer in background thread
    consumer_thread = threading.Thread(target=kafka_consumer_loop, daemon=True)
    consumer_thread.start()

    # Initialize producer
    get_kafka_producer()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global consumer, producer
    if consumer:
        consumer.close()
    if producer:
        producer.close()
    logger.info("Kafka connections closed")
    """Convert fraud score to risk level."""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "VERY_LOW"

@app.on_event("startup")
async def startup_event():
    """Initialize the fraud detection model."""
    global isolation_forest
    logger.info("Initializing fraud detection model...")

    # Train on synthetic data for demo
    np.random.seed(42)
    n_samples = 100
    vectors = []

    for _ in range(n_samples):
        # Generate synthetic transaction vectors
        amount = np.random.exponential(100)  # Most transactions small
        time_of_day = np.random.uniform(0, 1)
        merchant_hash = np.random.uniform(0, 1)
        device_hash = np.random.uniform(0, 1)
        country_hash = np.random.uniform(0, 1)
        risk_flag = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% risky

        vector = np.array([amount, time_of_day, merchant_hash, device_hash, country_hash, risk_flag])
        # L2 normalize
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)

    X = np.array(vectors)
    isolation_forest = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    isolation_forest.fit(X)

# Pydantic models
class VectorData(BaseModel):
    transaction_id: str
    vector: List[float]
    metadata: Optional[Dict] = None
    model_version: Optional[str] = None  # For A/B testing

class DetectionResult(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_score: float
    risk_level: str
    fraud_probability: float
    confidence: float
    model_used: str
    similar_transactions: List[str]
    insights: List[str]
    timestamp: str

class ModelUpdateData(BaseModel):
    model_version: str
    model_type: str
    model_data: Dict  # Serialized model data
    metadata: Optional[Dict] = None

def process_embedding_from_kafka(embedding_data: Dict):
    """Process an embedding from Kafka and run fraud detection."""
    try:
        transaction_id = embedding_data['transaction_id']
        vector = np.array(embedding_data['vector'])
        metadata = embedding_data.get('metadata', {})
        model_version = embedding_data.get('model_version', active_model_version)

        # Run fraud detection
        fraud_result = detect_fraud(transaction_id, vector, metadata, model_version)

        # Publish fraud result to Kafka for alert service
        kafka_producer = get_kafka_producer()
        if kafka_producer:
            kafka_producer.send(
                FRAUD_TOPIC,
                value=fraud_result,
                key=transaction_id
            )
            logger.info(f"Published fraud result for {transaction_id} to {FRAUD_TOPIC}")

        logger.info(f"Processed fraud detection for {transaction_id}: score={fraud_result['fraud_score']:.3f}")

    except Exception as e:
        logger.error(f"Error processing embedding {embedding_data.get('transaction_id', 'unknown')}: {e}")

def process_model_update_from_kafka(update_data: Dict):
    """Process model updates from Kafka for real-time model updates."""
    try:
        model_version = update_data['model_version']
        model_type = update_data['model_type']
        model_data = update_data['model_data']

        logger.info(f"Received model update: {model_version} ({model_type})")

        # In production, this would deserialize and update the model
        # For now, just log the update
        if model_type == 'ensemble':
            # Trigger ensemble model retraining
            initialize_models()
            logger.info(f"Retrained ensemble models for version {model_version}")

        # Update active model version if specified
        if update_data.get('set_active', False):
            global active_model_version
            active_model_version = model_version
            logger.info(f"Set active model version to {model_version}")

    except Exception as e:
        logger.error(f"Error processing model update: {e}")

def kafka_consumer_loop():
    """Background loop to consume embeddings and model updates from Kafka."""
    logger.info("Starting Kafka consumer loop...")
    consumer = get_kafka_consumer()
    if not consumer:
        logger.error("Kafka consumer not available, exiting loop")
        return

    try:
        for message in consumer:
            topic = message.topic
            data = message.value
            logger.info(f"Received message from topic {topic}")

            if topic == EMBEDDING_TOPIC:
                process_embedding_from_kafka(data)
            elif topic == MODEL_UPDATE_TOPIC:
                process_model_update_from_kafka(data)

    except Exception as e:
        logger.error(f"Kafka consumer loop error: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("=== CipherGuard Advanced Fraud Detection Service Starting ===")

    # Initialize ML models
    initialize_models()

    # Start Kafka consumer in background thread
    consumer_thread = threading.Thread(target=kafka_consumer_loop, daemon=True)
    consumer_thread.start()

    # Initialize producer
    get_kafka_producer()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global consumer, producer
    if consumer:
        consumer.close()
    if producer:
        producer.close()
    logger.info("Kafka connections closed")

@app.post("/detect", response_model=DetectionResult)
async def detect_fraud_endpoint(vector_data: VectorData):
    """Run advanced fraud detection on transaction vector."""
    try:
        if not models:
            raise HTTPException(status_code=503, detail="Models not initialized")

        vector = np.array(vector_data.vector)
        model_version = vector_data.model_version or active_model_version

        # Run fraud detection
        result = detect_fraud(
            vector_data.transaction_id,
            vector,
            vector_data.metadata,
            model_version
        )

        logger.info(f"Detection result - ID: {vector_data.transaction_id}, Score: {result['fraud_score']:.3f}, Risk: {result['risk_level']}")

        return DetectionResult(**result)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "trained" if models else "not_trained"
    active_version = active_model_version

    return {
        "service": "fraud-detection",
        "status": "healthy",
        "model_status": model_status,
        "active_model_version": active_version,
        "available_models": list(models.keys()) if models else [],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/retrain")
async def retrain_models():
    """Trigger model retraining."""
    try:
        logger.info("Retraining ensemble models...")
        initialize_models()
        return {
            "status": "retraining_completed",
            "models_retrained": list(models.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models and their performance metrics."""
    if not models:
        return {"models": [], "active_version": active_model_version}

    model_info = []
    for name, model in models.items():
        # Mock performance metrics (would be tracked in production)
        metrics = {
            "accuracy": 0.85 + np.random.uniform(-0.05, 0.05),
            "precision": 0.80 + np.random.uniform(-0.05, 0.05),
            "recall": 0.75 + np.random.uniform(-0.05, 0.05),
            "auc": 0.88 + np.random.uniform(-0.05, 0.05)
        }

        model_info.append({
            "name": name,
            "type": type(model).__name__,
            "metrics": metrics,
            "is_active": name == active_model_version
        })

    return {
        "models": model_info,
        "active_version": active_model_version,
        "ensemble_weights": {
            'isolation_forest': 0.2,
            'random_forest': 0.3,
            'xgboost': 0.3,
            'neural_network': 0.2
        }
    }

@app.post("/set-active-model")
async def set_active_model(model_data: Dict):
    """Set the active model version for A/B testing."""
    try:
        model_version = model_data.get('model_version')
        if model_version not in models:
            raise HTTPException(status_code=400, detail=f"Model version {model_version} not found")

        global active_model_version
        active_model_version = model_version

        logger.info(f"Set active model version to {model_version}")

        return {
            "status": "active_model_updated",
            "active_version": active_model_version,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics for fraud detection service."""
    if metrics_collector:
        return metrics_collector.get_metrics()
    else:
        return {"error": "Metrics collector not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

"""
Main FastAPI Application
Real-time fraud detection API with encrypted vector storage
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import logging
from datetime import datetime
import os
from sklearn.ensemble import IsolationForest

from .feature_extraction import extract_transaction_vector
from .cyborg_client import get_cyborg_client
from .cyborg_shim import get_cyborg_shim

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CipherGuard Fraud Detection API",
    description="Encrypted real-time fraud detection using CyborgDB",
    version="0.1.0"
)

# Initialize components
cyborg_client = None
cyborg_shim = get_cyborg_shim()  # Fallback mock
use_sdk = False
isolation_forest = None  # Will be initialized after collecting training data


# ============ Pydantic Models ============

class Transaction(BaseModel):
    """Transaction data model."""
    amount: float
    merchant: str
    device: str
    country: str
    timestamp: Optional[str] = None
    customer_id: Optional[str] = None


class DetectionResult(BaseModel):
    """Fraud detection result."""
    transaction_id: str
    is_fraud: bool
    fraud_score: float
    risk_level: str
    similar_transactions: List[str]
    timestamp: str


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    cyborg_vectors_count: int
    model_status: str
    timestamp: str


class FeedbackData(BaseModel):
    """User feedback for model retraining."""
    transaction_id: str
    was_fraud: bool
    feedback_text: Optional[str] = None


# ============ Helper Functions ============

def compute_fraud_score(vector: np.ndarray,
                       knn_results: List,
                       anomaly_score: float = 0.5) -> float:
    """Compute fraud score from multiple signals."""
    if not knn_results:
        return 0.5
    
    avg_distance = np.mean([dist for _, dist in knn_results])
    distance_score = min(avg_distance, 1.0)
    fraud_score = (anomaly_score * 0.4 + distance_score * 0.6)
    return float(fraud_score)


def get_risk_level(fraud_score: float) -> str:
    """Map fraud score to risk level."""
    if fraud_score < 0.3:
        return "LOW"
    elif fraud_score < 0.6:
        return "MEDIUM"
    elif fraud_score < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


# ============ API Endpoints ============

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global cyborg_client, use_sdk
    
    logger.info("=== CipherGuard Fraud Detection System Starting ===")
    
    try:
        cyborg_client = await get_cyborg_client()
        if cyborg_client.cyborg_client:
            use_sdk = True
            logger.info("✅ CyborgDB SDK initialized")
        else:
            logger.warning("⚠️  CyborgDB SDK not available, using local shim")
            use_sdk = False
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize CyborgDB SDK: {e}, using local shim")
        use_sdk = False
    
    logger.info(f"Mode: {'CyborgDB SDK' if use_sdk else 'Local Shim'}")
    logger.info(f"Vectors in store: {cyborg_shim.count()}")
    logger.info("API ready for requests")


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    model_status = "trained" if isolation_forest is not None else "not_trained"
    
    return HealthStatus(
        status="operational",
        cyborg_vectors_count=cyborg_shim.count(),
        model_status=model_status,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/detect", response_model=DetectionResult)
async def detect_fraud(transaction: Transaction):
    """Main fraud detection endpoint."""
    try:
        transaction_id = f"txn_{datetime.utcnow().timestamp()}"
        logger.info(f"Processing transaction: {transaction_id}")
        vector = extract_transaction_vector(transaction.dict())
        
        metadata = {
            "customer_id": transaction.customer_id,
            "timestamp": transaction.timestamp or datetime.utcnow().isoformat(),
            "merchant": transaction.merchant,
            "amount": transaction.amount
        }
        
        if use_sdk and cyborg_client:
            await cyborg_client.insert_transaction_vector(transaction_id, vector, metadata)
        else:
            cyborg_shim.insert(transaction_id, vector, metadata)
        
        if use_sdk and cyborg_client:
            knn_results = await cyborg_client.search_similar_vectors(vector, k=5)
        else:
            knn_results = cyborg_shim.search(vector, k=5)
        
        similar_ids = [tid for tid, _ in knn_results]
        
        if isolation_forest is not None:
            vector_2d = vector.reshape(1, -1)
            anomaly_pred = isolation_forest.predict(vector_2d)[0]
            anomaly_score = 1.0 if anomaly_pred == -1 else 0.2
        else:
            anomaly_score = 0.5
        
        fraud_score = compute_fraud_score(vector, knn_results, anomaly_score)
        is_fraud = fraud_score > 0.6
        risk_level = get_risk_level(fraud_score)
        
        logger.info(f"Detection result - Score: {fraud_score:.3f}, Risk: {risk_level}")
        
        return DetectionResult(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            fraud_score=fraud_score,
            risk_level=risk_level,
            similar_transactions=similar_ids[:3],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackData):
    """Submit feedback for model retraining."""
    try:
        logger.info(f"Received feedback for {feedback.transaction_id}: fraud={feedback.was_fraud}")
        return {
            "status": "feedback_received",
            "transaction_id": feedback.transaction_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Trigger model retraining."""
    try:
        global isolation_forest
        logger.info("Starting model retraining...")
        
        vectors = []
        for tid, vec in cyborg_shim.vectors.items():
            vectors.append(vec)
        
        if len(vectors) < 10:
            logger.warning(f"Insufficient data for training: {len(vectors)} vectors")
            return {
                "status": "skipped",
                "reason": "Insufficient training data (need >= 10 vectors)",
                "vectors_count": len(vectors)
            }
        
        X = np.array(vectors)
        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        isolation_forest.fit(X)
        
        logger.info(f"Model trained on {len(vectors)} vectors")
        
        return {
            "status": "training_started",
            "vectors_used": len(vectors),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = cyborg_shim.get_stats()
        stats["model_trained"] = isolation_forest is not None
        stats["timestamp"] = datetime.utcnow().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    backend = "CyborgDB SDK" if use_sdk else "Local Shim"
    
    return {
        "name": "CipherGuard Fraud Detection API",
        "version": "0.1.0",
        "description": "Encrypted real-time fraud detection using CyborgDB",
        "backend": backend,
        "endpoints": {
            "POST /detect": "Analyze transaction for fraud",
            "POST /feedback": "Submit analyst feedback for retraining",
            "POST /train": "Trigger model retraining",
            "GET /stats": "Get system statistics",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

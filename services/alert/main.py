"""
Alert Service - Handles fraud alerts and audit logging
Microservice for managing alerts, notifications, and audit trails with Kafka streaming
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime
import os
import asyncio
import threading
from kafka import KafkaConsumer
from cryptography.fernet import Fernet
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CipherGuard Alert Service",
    description="Fraud alerts and audit logging with Kafka streaming",
    version="0.1.0"
)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
FRAUD_TOPIC = 'transaction.fraud'

# Encryption setup (Phase 2: Secure Data Pipeline)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'your-32-char-encryption-key-here!')
cipher = Fernet(base64.urlsafe_b64encode(ENCRYPTION_KEY.encode()[:32].ljust(32, b'\0')))

# Global Kafka consumer
consumer = None

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data using Fernet symmetric encryption."""
    return cipher.decrypt(encrypted_data.encode()).decode()

def get_kafka_consumer():
    """Get or create Kafka consumer singleton."""
    global consumer
    if consumer is None:
        try:
            consumer = KafkaConsumer(
                FRAUD_TOPIC,
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                group_id='alert-service',
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

def process_fraud_result_from_kafka(fraud_data: Dict):
    """Process a fraud result from Kafka and create alerts."""
    try:
        transaction_id = fraud_data['transaction_id']

        # Create alert if fraudulent
        if fraud_data.get('is_fraud', False):
            alert_id = f"alert_{len(alerts_store) + 1}"

            alert_entry = {
                "alert_id": alert_id,
                "transaction_id": transaction_id,
                "fraud_score": fraud_data['fraud_score'],
                "risk_level": fraud_data['risk_level'],
                "is_fraud": fraud_data['is_fraud'],
                "customer_id": fraud_data.get('customer_id'),
                "amount": fraud_data.get('amount'),
                "merchant": fraud_data.get('merchant'),
                "timestamp": fraud_data['timestamp'],
                "created_at": datetime.now().isoformat()
            }

            alerts_store.append(alert_entry)

            # Log to audit trail
            audit_entry = {
                "event": "fraud_alert_created",
                "transaction_id": transaction_id,
                "alert_id": alert_id,
                "fraud_score": fraud_data['fraud_score'],
                "risk_level": fraud_data['risk_level'],
                "timestamp": datetime.now().isoformat()
            }
            audit_log.append(audit_entry)

            logger.warning(f"🚨 FRAUD ALERT created for {transaction_id}: score={fraud_data['fraud_score']:.3f}, risk={fraud_data['risk_level']}")
        else:
            # Log legitimate transaction
            audit_entry = {
                "event": "transaction_verified",
                "transaction_id": transaction_id,
                "fraud_score": fraud_data['fraud_score'],
                "risk_level": fraud_data['risk_level'],
                "timestamp": datetime.now().isoformat()
            }
            audit_log.append(audit_entry)

            logger.info(f"✅ Transaction {transaction_id} verified as legitimate: score={fraud_data['fraud_score']:.3f}")

    except Exception as e:
        logger.error(f"Error processing fraud result {fraud_data.get('transaction_id', 'unknown')}: {e}")

def kafka_consumer_loop():
    """Background loop to consume fraud results from Kafka."""
    logger.info("Starting Kafka consumer loop...")
    consumer = get_kafka_consumer()
    if not consumer:
        logger.error("Kafka consumer not available, exiting loop")
        return

    try:
        for message in consumer:
            fraud_data = message.value
            logger.info(f"Received fraud result from Kafka: {fraud_data['transaction_id']}")
            process_fraud_result_from_kafka(fraud_data)
    except Exception as e:
        logger.error(f"Kafka consumer loop error: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("=== CipherGuard Alert Service Starting ===")

    # Start Kafka consumer in background thread
    consumer_thread = threading.Thread(target=kafka_consumer_loop, daemon=True)
    consumer_thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global consumer
    if consumer:
        consumer.close()
    logger.info("Kafka consumer closed")

# Pydantic models
class AlertData(BaseModel):
    transaction_id: str
    fraud_score: float
    risk_level: str
    is_fraud: bool
    customer_id: Optional[str] = None
    amount: Optional[float] = None
    merchant: Optional[str] = None
    timestamp: str

class FeedbackData(BaseModel):
    transaction_id: str
    was_fraud: bool
    analyst_id: str
    comments: Optional[str] = None
    timestamp: str

class AlertResponse(BaseModel):
    alert_id: str
    status: str
    timestamp: str

# In-memory stores for demo (would be database in production)
alerts_store = []
feedback_store = []
audit_log = []

@app.post("/alert", response_model=AlertResponse)
async def create_alert(alert: AlertData):
    """Create a fraud alert."""
    try:
        alert_id = f"alert_{datetime.utcnow().timestamp()}"

        alert_record = {
            "alert_id": alert_id,
            "transaction_id": alert.transaction_id,
            "fraud_score": alert.fraud_score,
            "risk_level": alert.risk_level,
            "is_fraud": alert.is_fraud,
            "customer_id": alert.customer_id,
            "amount": alert.amount,
            "merchant": alert.merchant,
            "timestamp": alert.timestamp,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active"
        }

        alerts_store.append(alert_record)

        # Log to audit trail
        audit_log.append({
            "event": "alert_created",
            "alert_id": alert_id,
            "transaction_id": alert.transaction_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        # In production, this would:
        # - Send email/SMS notifications
        # - Create tickets in incident management system
        # - Trigger automated responses

        logger.info(f"Alert created: {alert_id} for transaction {alert.transaction_id}")

        return AlertResponse(
            alert_id=alert_id,
            status="created",
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Alert creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackData):
    """Submit analyst feedback for model improvement."""
    try:
        feedback_record = {
            "transaction_id": feedback.transaction_id,
            "was_fraud": feedback.was_fraud,
            "analyst_id": feedback.analyst_id,
            "comments": feedback.comments,
            "timestamp": feedback.timestamp,
            "submitted_at": datetime.utcnow().isoformat()
        }

        feedback_store.append(feedback_record)

        # Log to audit trail
        audit_log.append({
            "event": "feedback_submitted",
            "transaction_id": feedback.transaction_id,
            "analyst_id": feedback.analyst_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        logger.info(f"Feedback submitted for transaction: {feedback.transaction_id}")

        return {"status": "feedback_received", "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(status: Optional[str] = None, limit: int = 10):
    """Get alerts with optional filtering."""
    alerts = alerts_store

    if status:
        alerts = [a for a in alerts if a["status"] == status]

    return {
        "alerts": alerts[-limit:],
        "total": len(alerts),
        "filtered": len(alerts[-limit:])
    }

@app.get("/audit")
async def get_audit_log(limit: int = 50):
    """Get audit log entries."""
    return {
        "audit_log": audit_log[-limit:],
        "total": len(audit_log)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": "alert",
        "status": "healthy",
        "alerts_count": len(alerts_store),
        "feedback_count": len(feedback_store),
        "audit_entries": len(audit_log),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

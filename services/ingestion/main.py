"""
Ingestion Service - Handles transaction data ingestion
Microservice for receiving and validating transaction data with Kafka streaming
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime
import os
from kafka import KafkaProducer
from cryptography.fernet import Fernet
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CipherGuard Ingestion Service",
    description="Transaction data ingestion and validation with Kafka streaming",
    version="0.1.0"
)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
TRANSACTION_TOPIC = 'transaction.ingest'

# Encryption setup (Phase 2: Secure Data Pipeline)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'your-32-char-encryption-key-here!')
cipher = Fernet(base64.urlsafe_b64encode(ENCRYPTION_KEY.encode()[:32].ljust(32, b'\0')))

# Global Kafka producer
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
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            producer = None
    return producer

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("=== CipherGuard Ingestion Service Starting ===")
    get_kafka_producer()  # Initialize Kafka producer

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global producer
    if producer:
        producer.close()
        logger.info("Kafka producer closed")

# Pydantic models
class Transaction(BaseModel):
    amount: float
    merchant: str
    device: str
    country: str
    timestamp: Optional[str] = None
    customer_id: Optional[str] = None
    transaction_id: Optional[str] = None

class IngestionResponse(BaseModel):
    status: str
    transaction_id: str
    timestamp: str
    validated: bool

# In-memory queue for demo (would be Kafka/Redis in production)
transaction_queue = []

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_transaction(transaction: Transaction, background_tasks: BackgroundTasks):
    """Ingest and validate a transaction."""
    try:
        # Generate transaction ID if not provided
        transaction_id = transaction.transaction_id or f"txn_{datetime.utcnow().timestamp()}"

        # Validate transaction data
        if transaction.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")

        if not transaction.merchant.strip():
            raise HTTPException(status_code=400, detail="Merchant is required")

        # Add timestamp if not provided
        timestamp = transaction.timestamp or datetime.utcnow().isoformat()

        # Create validated transaction
        validated_transaction = {
            "transaction_id": transaction_id,
            "amount": transaction.amount,
            "merchant": transaction.merchant,
            "device": transaction.device,
            "country": transaction.country,
            "timestamp": timestamp,
            "customer_id": transaction.customer_id,
            "ingested_at": datetime.utcnow().isoformat()
        }

        # Publish to Kafka (Phase 2: Secure Data Pipeline)
        kafka_producer = get_kafka_producer()
        if kafka_producer:
            try:
                # Encrypt sensitive data before sending
                encrypted_transaction = validated_transaction.copy()
                if encrypted_transaction.get("customer_id"):
                    encrypted_transaction["customer_id"] = encrypt_data(str(encrypted_transaction["customer_id"]))

                # Send to Kafka topic
                future = kafka_producer.send(
                    TRANSACTION_TOPIC,
                    value=encrypted_transaction,
                    key=transaction_id
                )
                # Wait for confirmation (optional, but good for reliability)
                future.get(timeout=10)
                logger.info(f"Published transaction {transaction_id} to Kafka topic {TRANSACTION_TOPIC}")

            except Exception as e:
                logger.error(f"Failed to publish to Kafka: {e}")
                # Fallback to in-memory queue if Kafka fails
                transaction_queue.append(validated_transaction)
        else:
            # Fallback if Kafka not available
            transaction_queue.append(validated_transaction)
            logger.warning("Kafka not available, using fallback queue")

        # Background task to process (would send to embedding service)
        background_tasks.add_task(process_transaction, validated_transaction)

        logger.info(f"Ingested transaction: {transaction_id}")

        return IngestionResponse(
            status="ingested",
            transaction_id=transaction_id,
            timestamp=timestamp,
            validated=True
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": "ingestion",
        "status": "healthy",
        "queue_size": len(transaction_queue),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/queue")
async def get_queue():
    """Get current transaction queue (for debugging)."""
    return {"queue": transaction_queue[-10:]}  # Last 10 transactions

async def process_transaction(transaction: Dict):
    """Process transaction (would send to embedding service via Kafka)."""
    logger.info(f"Processing transaction: {transaction['transaction_id']}")
    # In production, this would send to Kafka topic for embedding service
    # For now, just log
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

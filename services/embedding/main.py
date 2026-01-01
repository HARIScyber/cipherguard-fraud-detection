"""
Embedding Service - Handles feature extraction and vector embedding
Microservice for converting transactions to encrypted vectors with Kafka streaming
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import numpy as np
from datetime import datetime
import os
import asyncio
import threading
from kafka import KafkaConsumer, KafkaProducer
from cryptography.fernet import Fernet
import base64
import json

# Import shared feature extraction
import sys
sys.path.append('../../..')
from app.feature_extraction import extract_transaction_vector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CipherGuard Embedding Service",
    description="Transaction feature extraction and vector embedding with Kafka streaming",
    version="0.1.0"
)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
TRANSACTION_TOPIC = 'transaction.ingest'
EMBEDDING_TOPIC = 'transaction.embedded'

# Encryption setup (Phase 2: Secure Data Pipeline)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'your-32-char-encryption-key-here!')
cipher = Fernet(base64.urlsafe_b64encode(ENCRYPTION_KEY.encode()[:32].ljust(32, b'\0')))

# Global Kafka components
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
                TRANSACTION_TOPIC,
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                group_id='embedding-service',
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

def process_transaction_from_kafka(transaction_data: Dict):
    """Process a transaction from Kafka and create embedding."""
    try:
        transaction_id = transaction_data['transaction_id']

        # Decrypt sensitive data
        if transaction_data.get('customer_id'):
            try:
                transaction_data['customer_id'] = decrypt_data(transaction_data['customer_id'])
            except:
                logger.warning(f"Could not decrypt customer_id for {transaction_id}")

        # Extract features and create vector
        vector = extract_transaction_vector(transaction_data)

        # Store vector (in production, this would go to CyborgDB)
        vector_store[transaction_id] = {
            "vector": vector.tolist(),
            "timestamp": datetime.now().isoformat(),
            "metadata": transaction_data
        }

        # Publish embedding result to Kafka for fraud detection service
        kafka_producer = get_kafka_producer()
        if kafka_producer:
            embedding_result = {
                "transaction_id": transaction_id,
                "vector": vector.tolist(),
                "vector_dimension": len(vector),
                "timestamp": datetime.now().isoformat(),
                "metadata": transaction_data
            }

            kafka_producer.send(
                EMBEDDING_TOPIC,
                value=embedding_result,
                key=transaction_id
            )
            logger.info(f"Published embedding for {transaction_id} to {EMBEDDING_TOPIC}")

        logger.info(f"Processed transaction: {transaction_id}, vector dimension: {len(vector)}")

    except Exception as e:
        logger.error(f"Error processing transaction {transaction_data.get('transaction_id', 'unknown')}: {e}")

def kafka_consumer_loop():
    """Background loop to consume transactions from Kafka."""
    logger.info("Starting Kafka consumer loop...")
    consumer = get_kafka_consumer()
    if not consumer:
        logger.error("Kafka consumer not available, exiting loop")
        return

    try:
        for message in consumer:
            transaction_data = message.value
            logger.info(f"Received transaction from Kafka: {transaction_data['transaction_id']}")
            process_transaction_from_kafka(transaction_data)
    except Exception as e:
        logger.error(f"Kafka consumer loop error: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("=== CipherGuard Embedding Service Starting ===")

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

# Pydantic models
class TransactionData(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    device: str
    country: str
    timestamp: str
    customer_id: Optional[str] = None

class EmbeddingResponse(BaseModel):
    transaction_id: str
    vector: List[float]
    vector_dimension: int
    timestamp: str

# In-memory store for demo (would be CyborgDB in production)
vector_store = {}

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_transaction(transaction: TransactionData):
    """Extract features and create embedding vector."""
    try:
        # Convert to dict for feature extraction
        transaction_dict = transaction.dict()

        # Extract features
        vector = extract_transaction_vector(transaction_dict)

        # Store vector (would send to CyborgDB in production)
        vector_store[transaction.transaction_id] = {
            "vector": vector.tolist(),
            "metadata": {
                "customer_id": transaction.customer_id,
                "timestamp": transaction.timestamp,
                "merchant": transaction.merchant,
                "amount": transaction.amount
            },
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Embedded transaction: {transaction.transaction_id}")

        return EmbeddingResponse(
            transaction_id=transaction.transaction_id,
            vector=vector.tolist(),
            vector_dimension=len(vector),
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vectors/{transaction_id}")
async def get_vector(transaction_id: str):
    """Retrieve stored vector."""
    if transaction_id not in vector_store:
        raise HTTPException(status_code=404, detail="Vector not found")

    return vector_store[transaction_id]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "service": "embedding",
        "status": "healthy",
        "vectors_stored": len(vector_store),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/vectors")
async def list_vectors(limit: int = 10):
    """List stored vectors (for debugging)."""
    vectors = list(vector_store.keys())[-limit:]
    return {"vectors": vectors, "total": len(vector_store)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

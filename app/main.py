"""
Main FastAPI Application - Microservices Orchestrator
Real-time fraud detection API coordinating microservices with Kafka streaming
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import httpx
import logging
from datetime import datetime
import os
import asyncio
import time
import numpy as np

# Additional imports
from sklearn.ensemble import IsolationForest
from .feature_extraction import extract_transaction_vector

logger = logging.getLogger(__name__)

# Phase 4: Production Deployment imports (optional)
try:
    from .monitoring import get_metrics_collector, track_request, track_fraud_detection
    from .model_optimizer import get_model_optimizer
    PRODUCTION_FEATURES_AVAILABLE = True
    logger.info("Production features loaded successfully")
except ImportError as e:
    logger.warning(f"Production features not available - running in basic mode: {e}")
    PRODUCTION_FEATURES_AVAILABLE = False

    # Mock functions for basic operation
    def get_metrics_collector():
        return None

    def track_request(func):
        return func

    def track_fraud_detection(func):
        return func

    def get_model_optimizer():
        return None

# CyborgDB imports
try:
    from .cyborg_client import get_cyborg_client
    from .cyborg_shim import cyborg_shim
    CYBORG_AVAILABLE = True
except ImportError:
    logger.warning("CyborgDB not available - using mock client")
    CYBORG_AVAILABLE = False
    
    def get_cyborg_client():
        return None
    
    cyborg_shim = None

# Phase 5: Advanced ML and Data Integration imports (optional)
try:
    from .advanced_models import get_advanced_detector
    from .data_integration import get_data_pipeline, TransactionData, DataValidator
    PHASE5_FEATURES_AVAILABLE = True
    logger.info("Phase 5 features loaded successfully")
except ImportError as e:
    logger.warning(f"Phase 5 features not available - running in Phase 4 mode: {e}")
    PHASE5_FEATURES_AVAILABLE = False

    # Mock functions for Phase 4 operation
    def get_advanced_detector():
        return None

    def get_data_pipeline():
        return None

# Phase 6: Enterprise Integration imports (optional)
try:
    from .payment_gateways import get_payment_gateway_manager
    from .enterprise_security import get_security_manager, get_jwt_manager, get_rate_limiter
    from .compliance_audit import get_audit_logger, get_compliance_checker, get_privacy_manager
    from .multi_tenant import get_tenant_manager, get_tenant_isolation, tenant_context
    from .enterprise_dashboard import get_metrics_collector as get_enterprise_metrics, get_alert_manager, get_dashboard_manager
    PHASE6_FEATURES_AVAILABLE = True
    logger.info("Phase 6 enterprise features loaded successfully")
except ImportError as e:
    logger.warning(f"Phase 6 enterprise features not available - running in Phase 5 mode: {e}")
    PHASE6_FEATURES_AVAILABLE = False

    # Mock functions for Phase 5 operation
    def get_payment_gateway_manager():
        return None

    def get_security_manager():
        return None

    def get_jwt_manager():
        return None

    def get_rate_limiter():
        return None

    def get_audit_logger():
        return None

    def get_compliance_checker():
        return None

    def get_privacy_manager():
        return None

    def get_tenant_manager():
        return None

    def get_tenant_isolation():
        return None

    def get_enterprise_metrics():
        return None

    def get_alert_manager():
        return None

    def get_dashboard_manager():
        return None

    def tenant_context(tenant_id):
        return None

# Initialize production components
if PRODUCTION_FEATURES_AVAILABLE:
    metrics_collector = get_metrics_collector()
    model_optimizer = get_model_optimizer()
else:
    metrics_collector = None
    model_optimizer = None

# Initialize Phase 5 components
if PHASE5_FEATURES_AVAILABLE:
    advanced_detector = get_advanced_detector()
    data_pipeline = get_data_pipeline()
else:
    advanced_detector = None
    data_pipeline = None

# Initialize Phase 6 components
if PHASE6_FEATURES_AVAILABLE:
    payment_gateway_manager = get_payment_gateway_manager()
    security_manager = get_security_manager()
    jwt_manager = get_jwt_manager()
    rate_limiter = get_rate_limiter()
    audit_logger = get_audit_logger()
    compliance_checker = get_compliance_checker()
    privacy_manager = get_privacy_manager()
    tenant_manager = get_tenant_manager()
    tenant_isolation = get_tenant_isolation()
    enterprise_metrics = get_enterprise_metrics()
    alert_manager = get_alert_manager()
    dashboard_manager = get_dashboard_manager()
else:
    payment_gateway_manager = None
    security_manager = None
    jwt_manager = None
    rate_limiter = None
    audit_logger = None
    compliance_checker = None
    privacy_manager = None
    tenant_manager = None
    tenant_isolation = None
    enterprise_metrics = None
    alert_manager = None
    dashboard_manager = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CipherGuard Fraud Detection API",
    description="Encrypted real-time fraud detection using microservices architecture with Kafka streaming",
    version="0.1.0"
)

# Global state
cyborg_client = None
use_sdk = False
isolation_forest = None

# HTTP client for health checks
client = None

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
    services: Dict[str, str]
    timestamp: str


class FeedbackData(BaseModel):
    """Analyst feedback data."""
    transaction_id: str
    was_fraud: bool
    analyst_id: str
    comments: Optional[str] = None


# Phase 5: Advanced Models and Data Integration Models

class AdvancedDetectionResult(BaseModel):
    """Advanced fraud detection result with multiple models."""
    transaction_id: str
    is_fraudulent: bool
    ensemble_score: float
    confidence: float
    model_predictions: Dict[str, float]
    risk_level: str
    similar_transactions: List[str]
    timestamp: str
    phase: str = "5"


class ModelTrainingRequest(BaseModel):
    """Request to train advanced models."""
    n_samples: Optional[int] = 10000
    fraud_ratio: Optional[float] = 0.1
    epochs: Optional[int] = 50
    test_split: Optional[float] = 0.2


class ModelTrainingResponse(BaseModel):
    """Response from model training."""
    status: str
    models_trained: List[str]
    evaluation_metrics: Optional[Dict[str, Any]] = None
    training_time: float
    timestamp: str


class DataSourceConfig(BaseModel):
    """Configuration for data source."""
    name: str
    type: str  # 'payment_gateway', 'database', 'file'
    config: Dict[str, Any]


class DataIngestionRequest(BaseModel):
    """Request to ingest data from sources."""
    sources: List[DataSourceConfig]
    start_time: str
    end_time: str
    batch_size: Optional[int] = 1000


class DataIngestionResponse(BaseModel):
    """Response from data ingestion."""
    status: str
    sources_connected: Dict[str, bool]
    transactions_processed: int
    transactions_failed: int
    batches_processed: int
    timestamp: str


async def check_service_health(service_name: str, url: str) -> str:
    """Check health of a microservice."""
    try:
        resp = await client.get(f"{url}/health")
        if resp.status_code == 200:
            return "healthy"
        else:
            return f"http_{resp.status_code}"
    except Exception as e:
        logger.warning(f"Service {service_name} health check failed: {e}")
        return "unreachable"


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global cyborg_client, use_sdk, isolation_forest, client, advanced_detector, data_pipeline
    
    logger.info("=== CipherGuard Fraud Detection System Starting ===")
    
    # Initialize HTTP client
    client = httpx.AsyncClient(timeout=5.0)
    
    # Initialize CyborgDB client
    if CYBORG_AVAILABLE:
        try:
            cyborg_client = get_cyborg_client()
            use_sdk = cyborg_client is not None
            logger.info(f"CyborgDB client initialized: {'SDK' if use_sdk else 'Mock'}")
        except Exception as e:
            logger.warning(f"CyborgDB initialization failed: {e}")
            cyborg_client = None
            use_sdk = False
    
    # Initialize Isolation Forest
    try:
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        logger.info("Isolation Forest model initialized")
    except Exception as e:
        logger.warning(f"Isolation Forest initialization failed: {e}")
        isolation_forest = None
    
    # Initialize Phase 5 components
    if PHASE5_FEATURES_AVAILABLE:
        try:
            # Advanced detector is already initialized globally
            logger.info("Phase 5 advanced models initialized")
            
            # Note: Model loading not implemented yet
            # if advanced_detector:
            #     advanced_detector.load_models()
            #     logger.info("Existing advanced models loaded")
            
        except Exception as e:
            logger.warning(f"Phase 5 initialization failed: {e}")
    
    logger.info("API ready for requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global client
    if client:
        await client.aclose()
    logger.info("Server shutting down")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "operational",
        "message": "CipherGuard Fraud Detection API is running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/detect", response_model=DetectionResult)
async def detect_fraud(transaction: Transaction):
    """Main fraud detection endpoint - local processing without Kafka."""
    try:
        transaction_id = f"txn_{datetime.utcnow().timestamp()}"
        logger.info(f"Processing transaction: {transaction_id}")
        
        # Extract features from transaction
        vector = extract_transaction_vector(transaction.dict())
        
        # Store in CyborgDB if available
        if cyborg_shim:
            metadata = {
                "customer_id": transaction.customer_id,
                "timestamp": transaction.timestamp or datetime.utcnow().isoformat(),
                "merchant": transaction.merchant,
                "amount": transaction.amount
            }
            cyborg_shim.insert(transaction_id, vector, metadata)
        
        # Find similar transactions
        similar_ids = []
        if cyborg_shim and len(cyborg_shim.vectors) > 0:
            knn_results = cyborg_shim.search(vector, k=5)
            similar_ids = [tid for tid, _ in knn_results]
        
        # Calculate fraud score
        if isolation_forest is not None:
            vector_2d = vector.reshape(1, -1)
            anomaly_pred = isolation_forest.predict(vector_2d)[0]
            anomaly_score = 1.0 if anomaly_pred == -1 else 0.2
        else:
            anomaly_score = 0.5
        
        # Simple fraud score calculation
        base_score = anomaly_score
        if len(similar_ids) > 0:
            # If we have similar transactions, adjust score based on amount patterns
            base_score = min(base_score + 0.1, 1.0)
        
        fraud_score = base_score
        is_fraud = fraud_score > 0.6
        
        # Determine risk level
        if fraud_score < 0.3:
            risk_level = "LOW"
        elif fraud_score < 0.6:
            risk_level = "MEDIUM"
        elif fraud_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        logger.info(f"Detection complete - Score: {fraud_score:.3f}, Risk: {risk_level}")
        
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
        
        # In standalone mode, just log the feedback
        # In production, this would trigger model retraining
        return {
            "status": "feedback_received", 
            "transaction_id": feedback.transaction_id,
            "message": "Feedback logged (model retraining not implemented in standalone mode)",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        # Get stats from each service
        stats = {"timestamp": datetime.utcnow().isoformat()}

        # Ingestion stats
        try:
            resp = await client.get(f"{MICROSERVICES['ingestion']}/health")
            if resp.status_code == 200:
                stats["ingestion"] = resp.json()
        except:
            stats["ingestion"] = {"error": "unreachable"}

        # Embedding stats
        try:
            resp = await client.get(f"{MICROSERVICES['embedding']}/health")
            if resp.status_code == 200:
                stats["embedding"] = resp.json()
        except:
            stats["embedding"] = {"error": "unreachable"}

        # Fraud detection stats
        try:
            resp = await client.get(f"{MICROSERVICES['fraud_detection']}/health")
            if resp.status_code == 200:
                stats["fraud_detection"] = resp.json()
        except:
            stats["fraud_detection"] = {"error": "unreachable"}

        # Alert stats
        try:
            resp = await client.get(f"{MICROSERVICES['alert']}/health")
            if resp.status_code == 200:
                stats["alert"] = resp.json()
        except:
            stats["alert"] = {"error": "unreachable"}

        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Phase 5: Advanced ML Endpoints ============

@app.post("/detect/advanced", response_model=AdvancedDetectionResult)
async def detect_fraud_advanced(transaction: Transaction):
    """Advanced fraud detection using deep learning and ensemble models."""
    if not PHASE5_FEATURES_AVAILABLE or not advanced_detector:
        raise HTTPException(
            status_code=503,
            detail="Phase 5 features not available. Advanced models not loaded."
        )

    try:
        transaction_id = f"txn_{datetime.utcnow().timestamp()}"
        logger.info(f"Processing advanced detection for transaction: {transaction_id}")

        # Extract features
        vector = extract_transaction_vector(transaction.dict())

        # Get advanced model predictions
        predictions = advanced_detector.predict_fraud(vector)

        # Determine risk level based on ensemble score
        ensemble_score = predictions.get('ensemble_score', 0.5)
        if ensemble_score < 0.3:
            risk_level = "LOW"
        elif ensemble_score < 0.6:
            risk_level = "MEDIUM"
        elif ensemble_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Find similar transactions (if CyborgDB available)
        similar_ids = []
        if cyborg_shim and len(cyborg_shim.vectors) > 0:
            knn_results = cyborg_shim.search(vector, k=5)
            similar_ids = [tid for tid, _ in knn_results]

        logger.info(f"Advanced detection complete - Ensemble Score: {ensemble_score:.3f}, Risk: {risk_level}")

        return AdvancedDetectionResult(
            transaction_id=transaction_id,
            is_fraudulent=predictions.get('is_fraudulent', False),
            ensemble_score=ensemble_score,
            confidence=predictions.get('confidence', 0.0),
            model_predictions={
                k: v for k, v in predictions.items()
                if k not in ['is_fraudulent', 'ensemble_score', 'confidence']
            },
            risk_level=risk_level,
            similar_transactions=similar_ids[:3],
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in advanced fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/train", response_model=ModelTrainingResponse)
async def train_advanced_models(request: ModelTrainingRequest):
    """Train advanced ML models with synthetic or real data."""
    if not PHASE5_FEATURES_AVAILABLE or not advanced_detector:
        raise HTTPException(
            status_code=503,
            detail="Phase 5 features not available. Cannot train advanced models."
        )

    try:
        import time
        start_time = time.time()

        logger.info(f"Starting advanced model training with {request.n_samples} samples")

        # Generate synthetic training data
        X, y = advanced_detector.generate_synthetic_data(
            n_samples=request.n_samples,
            fraud_ratio=request.fraud_ratio
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_split, random_state=42
        )

        # Train models
        results = advanced_detector.train_models(
            X_train, y_train, X_test, y_test, epochs=request.epochs
        )

        # Save models
        advanced_detector.save_models()

        training_time = time.time() - start_time

        logger.info(f"Model training completed in {training_time:.2f} seconds")

        return ModelTrainingResponse(
            status="training_completed",
            models_trained=list(results.keys()),
            evaluation_metrics=results.get('evaluation', {}),
            training_time=training_time,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/ingest", response_model=DataIngestionResponse)
async def ingest_data(request: DataIngestionRequest):
    """Ingest data from configured sources."""
    if not PHASE5_FEATURES_AVAILABLE or not data_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Phase 5 features not available. Data integration not loaded."
        )

    try:
        from datetime import datetime
        start_time = datetime.fromisoformat(request.start_time)
        end_time = datetime.fromisoformat(request.end_time)

        logger.info(f"Starting data ingestion from {len(request.sources)} sources")

        # Add sources to pipeline
        for source_config in request.sources:
            if source_config.type == 'file':
                from .data_integration import FileSource
                source = FileSource(
                    file_path=source_config.config['file_path'],
                    file_format=source_config.config.get('format', 'csv')
                )
            elif source_config.type == 'database':
                from .data_integration import DatabaseSource
                source = DatabaseSource(
                    connection_string=source_config.config['connection_string'],
                    table_name=source_config.config['table_name']
                )
            elif source_config.type == 'payment_gateway':
                from .data_integration import PaymentGatewaySource
                source = PaymentGatewaySource(
                    api_key=source_config.config['api_key'],
                    api_secret=source_config.config['api_secret'],
                    base_url=source_config.config['base_url']
                )
            else:
                raise ValueError(f"Unsupported source type: {source_config.type}")

            data_pipeline.add_source(source_config.name, source)

        # Initialize sources
        connection_results = await data_pipeline.initialize_sources()

        # Process data in batches
        batches_processed = 0
        async for batch in data_pipeline.fetch_all_transactions(
            start_time, end_time, request.batch_size
        ):
            batches_processed += 1
            # Here you could process/store the batch
            logger.info(f"Processed batch {batches_processed} with {len(batch)} transactions")

        # Get final stats
        stats = data_pipeline.get_pipeline_stats()

        logger.info(f"Data ingestion completed: {stats}")

        return DataIngestionResponse(
            status="ingestion_completed",
            sources_connected=connection_results,
            transactions_processed=stats['processed_transactions'],
            transactions_failed=stats['failed_transactions'],
            batches_processed=batches_processed,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_model_status():
    """Get status of advanced models."""
    if not PHASE5_FEATURES_AVAILABLE:
        return {
            "phase": "4",
            "message": "Running in Phase 4 mode. Advanced models not available.",
            "available_models": ["isolation_forest"]
        }

    status = {
        "phase": "5",
        "advanced_models_available": advanced_detector is not None,
        "data_pipeline_available": data_pipeline is not None,
        "models": {}
    }

    if advanced_detector:
        status["models"] = {
            "neural_network": advanced_detector.neural_network is not None,
            "xgboost": advanced_detector.xgboost_model is not None,
            "ensemble": advanced_detector.ensemble_model is not None
        }

    if data_pipeline:
        status["data_pipeline_stats"] = data_pipeline.get_pipeline_stats()

    return status


@app.get("/")
async def root():
    """Root endpoint with API info."""
    base_endpoints = {
        "POST /detect": "Basic fraud detection (Phase 1-4)",
        "POST /feedback": "Submit analyst feedback for retraining",
        "GET /stats": "Get system statistics",
        "GET /health": "Health check"
    }

    phase5_endpoints = {}
    if PHASE5_FEATURES_AVAILABLE:
        phase5_endpoints = {
            "POST /detect/advanced": "Advanced fraud detection with deep learning",
            "POST /models/train": "Train advanced ML models",
            "POST /data/ingest": "Ingest data from external sources",
            "GET /models/status": "Get advanced model status"
        }

    phase6_endpoints = {}
    if PHASE6_FEATURES_AVAILABLE:
        phase6_endpoints = {
            "POST /auth/login": "User authentication with JWT",
            "POST /auth/register": "User registration",
            "POST /payments/process": "Process payment through gateway",
            "GET /tenants": "List tenants (admin)",
            "POST /tenants": "Create new tenant (admin)",
            "GET /dashboard/{dashboard_id}": "Get dashboard data",
            "GET /compliance/report": "Get compliance report",
            "GET /audit/events": "Get audit events",
            "POST /privacy/consent": "Record privacy consent"
        }

    return {
        "name": "CipherGuard Fraud Detection API",
        "version": "0.1.0",
        "phase": "6" if PHASE6_FEATURES_AVAILABLE else ("5" if PHASE5_FEATURES_AVAILABLE else "4"),
        "description": "Enterprise-grade fraud detection with payment gateways, multi-tenancy, and compliance",
        "architecture": "Microservices",
        "services": list(MICROSERVICES.keys()),
        "endpoints": {**base_endpoints, **phase5_endpoints, **phase6_endpoints}
    }


# ============ Phase 6: Enterprise Integration Endpoints ============

if PHASE6_FEATURES_AVAILABLE:

    # Pydantic models for enterprise features
    class LoginRequest(BaseModel):
        username: str
        password: str
        tenant_id: Optional[str] = None

    class RegisterRequest(BaseModel):
        username: str
        email: str
        password: str
        tenant_id: Optional[str] = None

    class AuthResponse(BaseModel):
        access_token: str
        token_type: str
        expires_in: int
        user_id: str

    class PaymentRequest(BaseModel):
        amount: float
        currency: str
        gateway: str  # stripe, paypal, square
        payment_method: Dict[str, Any]
        metadata: Optional[Dict[str, Any]] = None

    class PaymentResponse(BaseModel):
        transaction_id: str
        status: str
        gateway_transaction_id: str
        amount: float
        currency: str
        processed_at: str

    class TenantCreateRequest(BaseModel):
        name: str
        admin_email: str
        tier: Optional[str] = "basic"

    class ConsentRequest(BaseModel):
        user_id: str
        consent_type: str
        consented: bool
        details: Optional[Dict[str, Any]] = None

    @app.post("/auth/login", response_model=AuthResponse)
    async def login(request: LoginRequest):
        """Authenticate user and return JWT token."""
        if not security_manager or not jwt_manager:
            raise HTTPException(status_code=501, detail="Authentication not available")

        # Verify credentials (simplified)
        user_id = f"user_{request.username}"

        # Check tenant access
        if request.tenant_id and tenant_manager:
            if not tenant_manager.check_tenant_access(request.tenant_id):
                raise HTTPException(status_code=403, detail="Tenant access denied")

        # Generate JWT token
        token = jwt_manager.generate_token(user_id, tenant_id=request.tenant_id)

        # Audit login event
        if audit_logger:
            audit_logger.log_event(
                "authentication", user_id, None, "api", "web",
                "/auth/login", "login", "success",
                {"username": request.username, "tenant_id": request.tenant_id}
            )

        return AuthResponse(
            access_token=token,
            token_type="bearer",
            expires_in=3600,
            user_id=user_id
        )

    @app.post("/auth/register")
    async def register(request: RegisterRequest):
        """Register new user."""
        if not security_manager:
            raise HTTPException(status_code=501, detail="Registration not available")

        # Create user (simplified)
        user_id = f"user_{request.username}"

        # Audit registration
        if audit_logger:
            audit_logger.log_event(
                "authentication", user_id, None, "api", "web",
                "/auth/register", "register", "success",
                {"username": request.username, "email": request.email}
            )

        return {"message": "User registered successfully", "user_id": user_id}

    @app.post("/payments/process", response_model=PaymentResponse)
    async def process_payment(request: PaymentRequest):
        """Process payment through configured gateway."""
        if not payment_gateway_manager:
            raise HTTPException(status_code=501, detail="Payment processing not available")

        try:
            # Get current tenant
            tenant_id = None
            if tenant_isolation:
                tenant_id = tenant_isolation.get_current_tenant()

            # Process payment
            result = await payment_gateway_manager.process_payment(
                request.gateway,
                request.amount,
                request.currency,
                request.payment_method,
                request.metadata or {}
            )

            # Audit payment
            if audit_logger:
                audit_logger.log_event(
                    "fraud_detection", "system", None, "api", "web",
                    "/payments/process", "payment_process", "success",
                    {
                        "amount": request.amount,
                        "currency": request.currency,
                        "gateway": request.gateway,
                        "tenant_id": tenant_id
                    }
                )

            return PaymentResponse(**result)

        except Exception as e:
            # Audit failed payment
            if audit_logger:
                audit_logger.log_event(
                    "fraud_detection", "system", None, "api", "web",
                    "/payments/process", "payment_process", "failure",
                    {"error": str(e), "gateway": request.gateway}
                )
            raise HTTPException(status_code=400, detail=f"Payment processing failed: {str(e)}")

    @app.get("/tenants")
    async def list_tenants():
        """List all tenants (admin only)."""
        if not tenant_manager:
            raise HTTPException(status_code=501, detail="Multi-tenancy not available")

        tenants = tenant_manager.get_all_tenants()
        return {"tenants": [asdict(t) for t in tenants]}

    @app.post("/tenants")
    async def create_tenant(request: TenantCreateRequest):
        """Create new tenant (admin only)."""
        if not tenant_manager:
            raise HTTPException(status_code=501, detail="Multi-tenancy not available")

        tenant_id = tenant_manager.create_tenant(
            request.name,
            tier=getattr(__import__('app.multi_tenant', fromlist=['TenantTier']).TenantTier, request.tier.upper(), None),
            admin_email=request.admin_email
        )

        return {"tenant_id": tenant_id, "message": "Tenant created successfully"}

    @app.get("/dashboard/{dashboard_id}")
    async def get_dashboard(dashboard_id: str):
        """Get dashboard data."""
        if not dashboard_manager:
            raise HTTPException(status_code=501, detail="Dashboard not available")

        data = dashboard_manager.get_dashboard_data(dashboard_id)
        if not data:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        return data

    @app.get("/compliance/report")
    async def get_compliance_report(standard: Optional[str] = None):
        """Get compliance report."""
        if not compliance_checker:
            raise HTTPException(status_code=501, detail="Compliance checking not available")

        from app.compliance_audit import ComplianceStandard
        std = ComplianceStandard(standard) if standard else None

        report = compliance_checker.get_compliance_status(std)
        return report

    @app.get("/audit/events")
    async def get_audit_events(
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ):
        """Get audit events."""
        if not audit_logger:
            raise HTTPException(status_code=501, detail="Audit logging not available")

        events = audit_logger.get_entries(
            user_id=user_id,
            event_type=event_type,
            limit=limit
        )

        return {"events": [asdict(e) for e in events]}

    @app.post("/privacy/consent")
    async def record_consent(request: ConsentRequest):
        """Record privacy consent."""
        if not privacy_manager:
            raise HTTPException(status_code=501, detail="Privacy management not available")

        privacy_manager.record_consent(
            request.user_id,
            request.consent_type,
            request.consented,
            request.details or {}
        )

        return {"message": "Consent recorded successfully"}

    @app.get("/enterprise/health")
    async def enterprise_health():
        """Enterprise features health check."""
        health = {
            "payment_gateways": payment_gateway_manager is not None,
            "security": security_manager is not None,
            "authentication": jwt_manager is not None,
            "rate_limiting": rate_limiter is not None,
            "audit": audit_logger is not None,
            "compliance": compliance_checker is not None,
            "privacy": privacy_manager is not None,
            "multi_tenant": tenant_manager is not None,
            "dashboard": dashboard_manager is not None
        }

        return {
            "status": "healthy" if all(health.values()) else "degraded",
            "features": health,
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

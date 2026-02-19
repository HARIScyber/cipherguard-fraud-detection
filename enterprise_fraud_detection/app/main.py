"""
Main FastAPI Application with Advanced Fraud Detection API
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time
import asyncio
from contextlib import asynccontextmanager
import uvicorn
import traceback

# Database imports
from database.init_db import get_database
from database.repositories.transaction_repository import TransactionRepository
from database.repositories.fraud_prediction_repository import FraudPredictionRepository
from database.repositories.user_behavior_repository import UserBehaviorRepository
from database.repositories.feedback_repository import FeedbackRepository

# ML imports  
from ml.inference.fraud_predictor import FraudDetectionPredictor, PredictionResult
from ml.training.model_trainer import FraudModelTrainer

# Core imports
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.core.monitoring import MetricsCollector, PerformanceTracker

# Initialize configuration
settings = get_settings()
setup_logging()
logger = logging.getLogger(__name__)

# Global variables for startup/shutdown
fraud_predictor: Optional[FraudDetectionPredictor] = None
metrics_collector: Optional[MetricsCollector] = None
performance_tracker: Optional[PerformanceTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Fraud Detection API...")
    
    global fraud_predictor, metrics_collector, performance_tracker
    
    try:
        # Initialize database
        database = get_database()
        
        # Initialize ML predictor
        with database.get_session() as session:
            fraud_predictor = FraudDetectionPredictor(
                models_dir=settings.ml.models_dir,
                db_session=session
            )
        
        # Initialize monitoring
        metrics_collector = MetricsCollector()
        performance_tracker = PerformanceTracker()
        
        logger.info("Fraud Detection API startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")


# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Fraud Detection API",
    description="Advanced fraud detection system with ML ensemble and real-time analytics",
    version="2.0.0",
    docs_url="/docs" if settings.api.enable_docs else None,
    redoc_url="/redoc" if settings.api.enable_docs else None,
    lifespan=lifespan
)

# Security
security = HTTPBearer(auto_error=False)

# CORS middleware
if settings.api.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Trusted host middleware
if settings.api.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.api.trusted_hosts
    )


# Request Models
class TransactionRequest(BaseModel):
    """Request model for fraud detection."""
    transaction_id: Optional[str] = Field(None, description="Optional transaction ID")
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant: str = Field(..., description="Merchant name")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    device: str = Field(default="unknown", description="Device type (mobile/desktop/tablet)")
    country: str = Field(default="Unknown", description="Country code")
    payment_method: str = Field(default="card", description="Payment method")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()


class FeedbackRequest(BaseModel):
    """Request model for fraud feedback."""
    transaction_id: str = Field(..., description="Transaction ID")
    is_fraud: bool = Field(..., description="True if transaction was actually fraud")
    fraud_type: Optional[str] = Field(None, description="Type of fraud if applicable")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence in feedback")
    notes: Optional[str] = Field(None, description="Additional notes")


class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    model_types: List[str] = Field(default=['random_forest', 'xgboost', 'ensemble'])
    retrain_existing: bool = Field(default=False)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    use_recent_data_only: bool = Field(default=False)
    days_lookback: int = Field(default=30, ge=1, le=365)


# Response Models
class FraudDetectionResponse(BaseModel):
    """Response model for fraud detection."""
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    confidence: float
    risk_level: str
    individual_scores: Dict[str, float]
    risk_factors: List[str]
    shap_values: Optional[Dict[str, float]]
    processing_time_ms: float
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalyticsResponse(BaseModel):
    """Response model for analytics."""
    fraud_rate_24h: float
    fraud_rate_7d: float
    fraud_rate_30d: float
    total_transactions_24h: int
    total_transactions_7d: int
    total_transactions_30d: int
    fraud_amount_24h: float
    fraud_amount_7d: float
    fraud_amount_30d: float
    risk_distribution: Dict[str, int]
    top_fraud_merchants: List[Dict[str, Any]]
    top_risk_factors: List[Dict[str, Any]]
    model_performance: Dict[str, float]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "2.0.0"
    database_connected: bool
    models_loaded: List[str]
    model_count: int
    uptime_seconds: float


# Dependency functions
async def get_db_session():
    """Get database session dependency."""
    database = get_database()
    with database.get_session() as session:
        yield session


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    if not settings.api.require_auth:
        return True
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if credentials.credentials != settings.api.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track API requests for monitoring."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} - {process_time:.3f}s"
    )
    
    # Collect metrics
    if metrics_collector:
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=process_time * 1000
        )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


# API Endpoints
@app.post(
    "/api/v1/detect",
    response_model=FraudDetectionResponse,
    summary="Detect Transaction Fraud",
    description="Analyze transaction for fraud using ensemble ML models"
)
async def detect_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    session = Depends(get_db_session),
    _: bool = Depends(verify_api_key)
):
    """Detect fraud in a transaction."""
    
    if not fraud_predictor or not fraud_predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection models not available"
        )
    
    try:
        # Convert request to dict for processing
        transaction_data = transaction.dict()
        
        # Make prediction
        result = fraud_predictor.predict_fraud(
            transaction_data=transaction_data,
            include_explanations=True,
            include_feature_vector=False
        )
        
        # Store transaction and prediction in database
        background_tasks.add_task(
            store_transaction_and_prediction,
            session, transaction_data, result
        )
        
        # Update user behavior
        background_tasks.add_task(
            update_user_behavior,
            session, transaction.user_id, transaction_data, result
        )
        
        # Convert result to response
        response = FraudDetectionResponse(
            transaction_id=result.transaction_id,
            fraud_score=result.fraud_score,
            is_fraud=result.is_fraud,
            confidence=result.confidence,
            risk_level=result.risk_level,
            individual_scores=result.individual_scores,
            risk_factors=result.risk_factors,
            shap_values=result.shap_values,
            processing_time_ms=result.processing_time_ms,
            model_version=result.model_version
        )
        
        logger.info(f"Fraud detection completed for {result.transaction_id}: {result.fraud_score:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {e}")
        logger.debug(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fraud detection processing failed"
        )


@app.post(
    "/api/v1/feedback",
    summary="Submit Feedback",
    description="Submit feedback on fraud detection accuracy"
)
async def submit_feedback(
    feedback: FeedbackRequest,
    session = Depends(get_db_session),
    _: bool = Depends(verify_api_key)
):
    """Submit feedback on fraud detection accuracy."""
    
    try:
        # Store feedback in database
        feedback_repo = FeedbackRepository(session)
        
        feedback_record = feedback_repo.create_feedback(
            transaction_id=feedback.transaction_id,
            is_fraud=feedback.is_fraud,
            fraud_type=feedback.fraud_type,
            confidence=feedback.confidence,
            notes=feedback.notes
        )
        
        logger.info(f"Feedback submitted for transaction {feedback.transaction_id}: is_fraud={feedback.is_fraud}")
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback_id": feedback_record.id
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback"
        )


@app.get(
    "/api/v1/analytics",
    response_model=AnalyticsResponse,
    summary="Get Analytics",
    description="Retrieve fraud detection analytics and statistics"
)
async def get_analytics(
    session = Depends(get_db_session),
    _: bool = Depends(verify_api_key)
):
    """Get fraud detection analytics."""
    
    try:
        # Initialize repositories
        transaction_repo = TransactionRepository(session)
        prediction_repo = FraudPredictionRepository(session)
        
        # Calculate time periods
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # Get transaction statistics
        stats_24h = transaction_repo.get_fraud_statistics(day_ago, now)
        stats_7d = transaction_repo.get_fraud_statistics(week_ago, now)
        stats_30d = transaction_repo.get_fraud_statistics(month_ago, now)
        
        # Get risk distribution
        risk_dist = prediction_repo.get_risk_distribution(day_ago, now)
        
        # Get top fraud merchants
        fraud_merchants = transaction_repo.get_top_fraud_merchants(month_ago, now, limit=10)
        
        # Get top risk factors
        risk_factors = prediction_repo.get_top_risk_factors(month_ago, now, limit=10)
        
        # Get model performance (if available)
        model_performance = {}
        if fraud_predictor:
            model_info = fraud_predictor.get_model_info()
            for model_type, info in model_info.items():
                model_performance[model_type] = info.performance_metrics.get('f1_score', 0.0)
        
        analytics = AnalyticsResponse(
            fraud_rate_24h=stats_24h.get('fraud_rate', 0.0),
            fraud_rate_7d=stats_7d.get('fraud_rate', 0.0),
            fraud_rate_30d=stats_30d.get('fraud_rate', 0.0),
            total_transactions_24h=stats_24h.get('total_transactions', 0),
            total_transactions_7d=stats_7d.get('total_transactions', 0),
            total_transactions_30d=stats_30d.get('total_transactions', 0),
            fraud_amount_24h=stats_24h.get('fraud_amount', 0.0),
            fraud_amount_7d=stats_7d.get('fraud_amount', 0.0),
            fraud_amount_30d=stats_30d.get('fraud_amount', 0.0),
            risk_distribution=risk_dist,
            top_fraud_merchants=fraud_merchants,
            top_risk_factors=risk_factors,
            model_performance=model_performance
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and system health status"
)
async def health_check(session = Depends(get_db_session)):
    """Check system health."""
    
    try:
        # Test database connection
        try:
            session.execute("SELECT 1")
            db_connected = True
        except:
            db_connected = False
        
        # Get model status
        models_loaded = []
        model_count = 0
        if fraud_predictor:
            model_info = fraud_predictor.get_model_info()
            models_loaded = list(model_info.keys())
            model_count = len(models_loaded)
        
        # Calculate uptime (simplified)
        uptime = time.time()  # Would need proper startup time tracking
        
        status_val = "healthy" if db_connected and model_count > 0 else "degraded"
        
        health = HealthResponse(
            status=status_val,
            database_connected=db_connected,
            models_loaded=models_loaded,
            model_count=model_count,
            uptime_seconds=uptime
        )
        
        return health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@app.get(
    "/api/v1/models",
    summary="Get Model Information",
    description="Retrieve information about loaded ML models"
)
async def get_model_info(_: bool = Depends(verify_api_key)):
    """Get information about loaded models."""
    
    if not fraud_predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection system not available"
        )
    
    try:
        model_info = fraud_predictor.get_model_info()
        health_status = fraud_predictor.get_health_status()
        
        return {
            "models": {
                model_type: {
                    "model_type": info.model_type,
                    "model_version": info.model_version,
                    "feature_count": len(info.feature_names),
                    "performance_metrics": info.performance_metrics,
                    "created_at": info.created_at
                }
                for model_type, info in model_info.items()
            },
            "system_status": health_status
        }
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@app.post(
    "/api/v1/train",
    summary="Train Models",
    description="Trigger model training with latest data"
)
async def train_models(
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    session = Depends(get_db_session),
    _: bool = Depends(verify_api_key)
):
    """Trigger model training."""
    
    try:
        # Add training task to background
        background_tasks.add_task(
            train_models_task,
            session,
            training_request.model_types,
            training_request.retrain_existing,
            training_request.test_size,
            training_request.use_recent_data_only,
            training_request.days_lookback
        )
        
        return {
            "status": "success",
            "message": "Model training started",
            "model_types": training_request.model_types
        }
        
    except Exception as e:
        logger.error(f"Model training initiation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start model training"
        )


# Background tasks
async def store_transaction_and_prediction(
    session, 
    transaction_data: Dict[str, Any], 
    result: PredictionResult
):
    """Store transaction and prediction in database."""
    
    try:
        # Store transaction
        transaction_repo = TransactionRepository(session)
        transaction_record = transaction_repo.create_transaction(
            transaction_id=result.transaction_id,
            user_id=transaction_data['user_id'],
            amount=transaction_data['amount'],
            merchant=transaction_data['merchant'],
            timestamp=transaction_data['timestamp'],
            device=transaction_data.get('device', 'unknown'),
            country=transaction_data.get('country', 'Unknown'),
            payment_method=transaction_data.get('payment_method', 'card'),
            merchant_category=transaction_data.get('merchant_category'),
            ip_address=transaction_data.get('ip_address'),
            session_id=transaction_data.get('session_id')
        )
        
        # Store prediction
        prediction_repo = FraudPredictionRepository(session)
        prediction_repo.create_prediction(
            transaction_id=result.transaction_id,
            fraud_score=result.fraud_score,
            is_fraud=result.is_fraud,
            risk_level=result.risk_level,
            model_version=result.model_version,
            processing_time_ms=result.processing_time_ms,
            individual_scores=result.individual_scores,
            risk_factors=result.risk_factors,
            shap_values=result.shap_values
        )
        
        session.commit()
        logger.debug(f"Stored transaction and prediction for {result.transaction_id}")
        
    except Exception as e:
        logger.error(f"Failed to store transaction data: {e}")
        session.rollback()


async def update_user_behavior(
    session,
    user_id: str,
    transaction_data: Dict[str, Any],
    result: PredictionResult
):
    """Update user behavior patterns."""
    
    try:
        behavior_repo = UserBehaviorRepository(session)
        behavior_repo.update_user_behavior(
            user_id=user_id,
            transaction_amount=transaction_data['amount'],
            merchant=transaction_data['merchant'],
            device=transaction_data.get('device', 'unknown'),
            country=transaction_data.get('country', 'Unknown'),
            fraud_score=result.fraud_score,
            timestamp=transaction_data['timestamp']
        )
        
        session.commit()
        logger.debug(f"Updated user behavior for {user_id}")
        
    except Exception as e:
        logger.error(f"Failed to update user behavior: {e}")
        session.rollback()


async def train_models_task(
    session,
    model_types: List[str],
    retrain_existing: bool,
    test_size: float,
    use_recent_data_only: bool,
    days_lookback: int
):
    """Background task for model training."""
    
    try:
        logger.info(f"Starting model training: {model_types}")
        
        trainer = FraudModelTrainer(session)
        
        # Train models
        results = trainer.train_models(
            model_types=model_types,
            retrain_existing=retrain_existing,
            test_size=test_size,
            use_recent_data_only=use_recent_data_only,
            days_lookback=days_lookback
        )
        
        logger.info(f"Model training completed: {results}")
        
        # Reload models in predictor
        global fraud_predictor
        if fraud_predictor:
            fraud_predictor._load_models()
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc} - {request.url}")
    logger.debug(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "status_code": 500}
    )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level=settings.logging.level.lower(),
        access_log=True
    )
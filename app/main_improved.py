"""
Improved main application with proper architecture patterns
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Import our improved modules
from .service_registry import services, init_services
from .enhanced_models import Transaction, DetectionResult, RiskLevel
from .security_middleware import (
    SecurityMiddleware, 
    InMemoryRateLimiter,
    validate_request_size_middleware,
    security_headers_middleware,
    audit_middleware
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionService:
    """Core fraud detection business logic."""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        
    async def initialize(self):
        """Initialize ML models and dependencies."""
        try:
            # Initialize isolation forest
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=0.1, 
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            
            # Initialize feature extractor
            from .feature_extraction import FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            
            logger.info("Fraud detection service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize fraud detection service: {e}")
            raise
            
    async def detect_fraud(self, transaction: Transaction) -> DetectionResult:
        """Core fraud detection logic with improved error handling."""
        start_time = datetime.utcnow()
        transaction_id = f"txn_{int(start_time.timestamp() * 1000)}"
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(transaction.dict())
            
            # Get predictions from multiple models
            fraud_score = await self._calculate_fraud_score(features, transaction)
            confidence = await self._calculate_confidence(features, transaction)
            risk_factors = await self._identify_risk_factors(transaction, fraud_score)
            
            # Determine risk level
            risk_level = self._determine_risk_level(fraud_score)
            
            # Find similar transactions
            similar_transactions = await self._find_similar_transactions(features)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return DetectionResult(
                transaction_id=transaction_id,
                is_fraud=fraud_score > 0.6,
                fraud_score=fraud_score,
                confidence=confidence,
                risk_level=risk_level,
                risk_factors=risk_factors,
                similar_transactions=similar_transactions,
                timestamp=start_time,
                processing_time_ms=processing_time,
                model_version="v1.0.0"
            )
            
        except Exception as e:
            logger.error(f"Fraud detection failed for transaction {transaction_id}: {e}")
            raise HTTPException(status_code=500, detail="Fraud detection service unavailable")
    
    async def _calculate_fraud_score(self, features, transaction: Transaction) -> float:
        """Calculate fraud score using ensemble of methods."""
        scores = []
        
        # Isolation Forest score
        if self.model is not None:
            isolation_score = self.model.decision_function(features.reshape(1, -1))[0]
            # Convert to 0-1 probability
            isolation_prob = max(0, min(1, (1 - isolation_score) / 2))
            scores.append(isolation_prob)
        
        # Rule-based scoring
        rule_score = await self._rule_based_scoring(transaction)
        scores.append(rule_score)
        
        # Velocity checks
        velocity_score = await self._velocity_scoring(transaction)  
        scores.append(velocity_score)
        
        # Weighted average
        weights = [0.4, 0.3, 0.3]
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, final_score))
    
    async def _rule_based_scoring(self, transaction: Transaction) -> float:
        """Rule-based fraud scoring."""
        score = 0.0
        
        # High amount transactions
        if transaction.amount > 10000:
            score += 0.3
        elif transaction.amount > 5000:
            score += 0.15
            
        # Off-hours transactions (simplified)
        if hasattr(transaction, 'timestamp') and transaction.timestamp:
            hour = transaction.timestamp.hour
            if hour < 6 or hour > 22:  # Night transactions
                score += 0.2
                
        # High-risk countries (simplified)  
        high_risk_countries = {'XX', 'YY', 'ZZ'}  # Replace with actual list
        if transaction.country in high_risk_countries:
            score += 0.25
            
        return min(1.0, score)
    
    async def _velocity_scoring(self, transaction: Transaction) -> float:
        """Velocity-based fraud scoring."""
        # This would check transaction velocity from database
        # For now, return baseline score
        return 0.1
    
    async def _calculate_confidence(self, features, transaction: Transaction) -> float:
        """Calculate model confidence score."""
        # Simplified confidence calculation
        # In production, this could use model uncertainty quantification
        base_confidence = 0.85
        
        # Reduce confidence for edge cases
        if transaction.amount > 50000:  # Very high amounts
            base_confidence -= 0.2
            
        return max(0.1, min(1.0, base_confidence))
    
    async def _identify_risk_factors(self, transaction: Transaction, fraud_score: float) -> list:
        """Identify specific risk factors."""
        factors = []
        
        if transaction.amount > 10000:
            factors.append("High transaction amount")
            
        if fraud_score > 0.8:
            factors.append("Anomalous transaction pattern")
            
        # Add more risk factor logic here
        
        return factors
    
    def _determine_risk_level(self, fraud_score: float) -> RiskLevel:
        """Determine risk level based on fraud score."""
        if fraud_score < 0.3:
            return RiskLevel.LOW
        elif fraud_score < 0.6:
            return RiskLevel.MEDIUM
        elif fraud_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    async def _find_similar_transactions(self, features) -> list:
        """Find similar historical transactions."""
        # This would query the vector database
        # For now, return empty list
        return []

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting CipherGuard Fraud Detection API...")
    
    try:
        # Initialize services
        init_services()
        
        # Initialize fraud detection service
        fraud_service = FraudDetectionService()
        await fraud_service.initialize()
        app.state.fraud_service = fraud_service
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("Shutting down CipherGuard API...")

# Create FastAPI app
app = FastAPI(
    title="CipherGuard Fraud Detection API",
    description="Production-grade fraud detection with ML and security best practices",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add security middleware
rate_limiter = InMemoryRateLimiter()  # Use Redis in production
security = SecurityMiddleware(rate_limiter, os.getenv("JWT_SECRET", "dev-secret"))

# Add middleware in correct order
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])
app.middleware("http")(security_headers_middleware)
app.middleware("http")(validate_request_size_middleware)
app.middleware("http")(audit_middleware)
app.middleware("http")(security.create_rate_limit_middleware())

# Health check endpoint
@app.get("/health")
async def health_check():
    """Enhanced health check with dependency status."""
    return {
        "status": "operational",
        "message": "CipherGuard Fraud Detection API is running", 
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "dependencies": {
            "cyborg_client": services.get_service("cyborg_client") is not None,
            "metrics": services.get_service("metrics") is not None,
            "security": services.get_service("security") is not None
        }
    }

# Main fraud detection endpoint
@app.post("/detect", response_model=DetectionResult)
async def detect_fraud(
    transaction: Transaction,
    request: Request
) -> DetectionResult:
    """Enhanced fraud detection with comprehensive error handling."""
    
    fraud_service = request.app.state.fraud_service
    
    try:
        result = await fraud_service.detect_fraud(transaction)
        
        # Log high-risk transactions
        if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.warning(
                f"High-risk transaction detected: {result.transaction_id}, "
                f"Score: {result.fraud_score:.3f}, Risk: {result.risk_level}"
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Fraud detection error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal fraud detection error"
        )

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """System metrics for monitoring."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": "TODO",  # Implement uptime tracking
        "request_count": "TODO",   # Implement request counter
        "error_rate": "TODO"       # Implement error tracking
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8001,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level="info"
    )
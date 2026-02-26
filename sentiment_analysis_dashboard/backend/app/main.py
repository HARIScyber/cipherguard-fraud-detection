"""
FastAPI application for Comment Sentiment Analysis Dashboard.
Production-ready sentiment analysis API with authentication and database integration.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime

from .database import engine, get_db
from .models import Base
from .routes.comments import router as comments_router
from .routes.health import router as health_router
from .routes.auth import router as auth_router
from .routes.fraud import router as fraud_router
from .services.sentiment_analyzer import SentimentAnalyzer
from .services.fraud_detector import get_fraud_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global sentiment analyzer instance
sentiment_analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Comment Sentiment Analysis API...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
    
    # Initialize ML model
    try:
        global sentiment_analyzer
        sentiment_analyzer = SentimentAnalyzer()
        await sentiment_analyzer.load_model()
        logger.info("Sentiment analysis model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        raise
    
    # Initialize fraud detector
    try:
        fraud_detector = get_fraud_detector()
        await fraud_detector.load_model()
        logger.info("Fraud detection model loaded successfully")
    except Exception as e:
        logger.warning(f"Fraud detector using rule-based mode: {e}")
    
    logger.info("Application startup completed")
    yield
    
    # Shutdown
    logger.info("Shutting down Comment Sentiment Analysis API...")

# Create FastAPI application
app = FastAPI(
    title="Comment Sentiment Analysis API",
    description="""
    ## Production-Ready Sentiment Analysis Dashboard API
    
    This API provides real-time comment sentiment analysis with:
    
    * **Real-time Analysis**: Instant sentiment classification for comments
    * **Confidence Scoring**: Probability scores for sentiment predictions  
    * **Historical Data**: Storage and retrieval of all analyzed comments
    * **Enterprise Security**: JWT authentication and secure endpoints
    * **Performance Optimized**: Fast predictions with cached ML models
    
    ### Authentication
    
    This API uses JWT Bearer token authentication. Include your token in the Authorization header:
    ```
    Authorization: Bearer <your-jwt-token>
    ```
    
    ### Rate Limiting
    
    API requests are rate-limited to ensure fair usage and system stability.
    """,
    version="1.0.0",
    contact={
        "name": "Sentiment Analysis Team",
        "email": "support@sentiment-dashboard.com"
    },
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])
app.include_router(comments_router, prefix="/api/v1", tags=["Comments"])
app.include_router(fraud_router, prefix="/api/v1", tags=["Fraud Detection"])
app.include_router(health_router, prefix="/api", tags=["Health"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Comment Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs",
        "health_check": "/api/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An internal server error occurred"
    )

# Dependency to get sentiment analyzer
async def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Dependency to get the global sentiment analyzer instance."""
    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not available"
        )
    return sentiment_analyzer

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
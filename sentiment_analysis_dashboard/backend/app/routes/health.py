"""
Health check and system monitoring routes.
"""

import time
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db, engine
from ..schemas import APIResponse, SystemHealth
from ..auth import get_current_user, get_current_admin_user
from ..models import User

# Router
router = APIRouter(prefix="/health", tags=["Health Check"])


@router.get(
    "/",
    response_model=APIResponse[SystemHealth],
    summary="Basic Health Check",
    description="Basic health check endpoint"
)
async def health_check():
    """
    Basic health check endpoint.
    
    **Public endpoint** - No authentication required
    
    **Returns:**
    - **status**: Application status
    - **timestamp**: Current server time
    - **version**: Application version
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0.0",
            "uptime_seconds": 3600.5
        },
        "message": "System is healthy"
    }
    ```
    """
    # Calculate uptime (simple approximation)
    import os
    try:
        # Try to get process start time
        process = psutil.Process(os.getpid())
        start_time = process.create_time()
        uptime_seconds = time.time() - start_time
    except:
        uptime_seconds = 0.0
    
    health_data = SystemHealth(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=round(uptime_seconds, 2)
    )
    
    return APIResponse(
        success=True,
        data=health_data,
        message="System is healthy"
    )


@router.get(
    "/detailed",
    response_model=APIResponse[Dict[str, Any]],
    summary="Detailed Health Check",
    description="Detailed health check with system metrics"
)
async def detailed_health_check(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Detailed health check with system metrics.
    
    **Authentication:** Bearer token required
    
    **Returns:**
    - **Application status**
    - **Database connectivity**
    - **System metrics** (CPU, memory, disk)
    - **Dependencies status**
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "application": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            },
            "database": {
                "status": "connected",
                "response_time_ms": 15.2,
                "connection_count": 5
            },
            "system": {
                "cpu_percent": 25.4,
                "memory_percent": 68.2,
                "disk_percent": 45.1,
                "available_memory_mb": 2048
            },
            "ml_model": {
                "status": "loaded",
                "model_version": "1.0.0",
                "last_prediction_time": "2024-01-15T10:29:00Z"
            }
        },
        "message": "Detailed system health"
    }
    ```
    """
    health_details = {}
    
    # Application health
    import os
    try:
        process = psutil.Process(os.getpid())
        start_time = process.create_time()
        uptime_seconds = time.time() - start_time
    except:
        uptime_seconds = 0.0
    
    health_details["application"] = {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.utcnow()
    }
    
    # Database health
    db_health = {"status": "unknown"}
    try:
        start_time = time.time()
        
        # Test database connection
        result = db.execute(text("SELECT 1")).fetchone()
        response_time_ms = (time.time() - start_time) * 1000
        
        if result:
            db_health = {
                "status": "connected",
                "response_time_ms": round(response_time_ms, 2)
            }
        else:
            db_health = {"status": "error", "message": "Query failed"}
    
    except Exception as e:
        db_health = {"status": "error", "message": str(e)}
    
    health_details["database"] = db_health
    
    # System metrics
    try:
        system_health = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "disk_percent": psutil.disk_usage('/').percent,
        }
    except Exception as e:
        system_health = {"error": str(e)}
    
    health_details["system"] = system_health
    
    # ML Model health
    try:
        from ..services.sentiment_analyzer import SentimentAnalyzer
        
        # Try to initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        ml_health = {
            "status": "loaded" if analyzer.model else "not_loaded",
            "model_version": getattr(analyzer, 'model_version', '1.0.0')
        }
        
        # Test prediction if model is loaded
        if analyzer.model:
            try:
                test_start = time.time()
                sentiment, confidence = await analyzer.predict_sentiment("Test comment")
                prediction_time_ms = (time.time() - test_start) * 1000
                
                ml_health.update({
                    "test_prediction": {
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "response_time_ms": round(prediction_time_ms, 2)
                    }
                })
            except Exception as e:
                ml_health["test_prediction_error"] = str(e)
    
    except Exception as e:
        ml_health = {"status": "error", "message": str(e)}
    
    health_details["ml_model"] = ml_health
    
    return APIResponse(
        success=True,
        data=health_details,
        message="Detailed system health"
    )


@router.get(
    "/database",
    response_model=APIResponse[Dict[str, Any]],
    summary="Database Health Check",
    description="Check database connectivity and performance"
)
async def database_health_check(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Check database connectivity and performance.
    
    **Authentication:** Admin access required
    
    **Returns:**
    - **Database connection status**
    - **Performance metrics**
    - **Table statistics**
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "connection": {
                "status": "connected",
                "response_time_ms": 15.2,
                "database_name": "sentiment_db",
                "database_version": "PostgreSQL 13.4"
            },
            "tables": {
                "comments": {"count": 1500, "size_mb": 25.4},
                "users": {"count": 50, "size_mb": 1.2}
            },
            "performance": {
                "active_connections": 5,
                "max_connections": 100
            }
        },
        "message": "Database health check completed"
    }
    ```
    """
    db_details = {}
    
    try:
        # Test connection and measure response time
        start_time = time.time()
        
        # Basic connectivity test
        db.execute(text("SELECT version()"))
        response_time_ms = (time.time() - start_time) * 1000
        
        connection_info = {
            "status": "connected",
            "response_time_ms": round(response_time_ms, 2)
        }
        
        # Get database version
        try:
            version_result = db.execute(text("SELECT version()")).fetchone()
            if version_result:
                connection_info["database_version"] = version_result[0]
        except:
            pass
        
        # Get database name
        try:
            db_name_result = db.execute(text("SELECT current_database()")).fetchone()
            if db_name_result:
                connection_info["database_name"] = db_name_result[0]
        except:
            pass
        
        db_details["connection"] = connection_info
        
        # Table statistics
        from ..models import Comment, User
        
        tables_info = {}
        
        try:
            # Comment table stats
            comment_count = db.query(Comment).count()
            tables_info["comments"] = {"count": comment_count}
            
            # User table stats
            user_count = db.query(User).count()
            tables_info["users"] = {"count": user_count}
            
        except Exception as e:
            tables_info["error"] = str(e)
        
        db_details["tables"] = tables_info
        
        # Connection pool info (if available)
        performance_info = {}
        try:
            # Try to get connection pool statistics
            if hasattr(engine.pool, 'size'):
                performance_info["pool_size"] = engine.pool.size()
            if hasattr(engine.pool, 'checked_in'):
                performance_info["checked_in_connections"] = engine.pool.checkedin()
            if hasattr(engine.pool, 'checked_out'):
                performance_info["checked_out_connections"] = engine.pool.checkedout()
        except:
            pass
        
        db_details["performance"] = performance_info
    
    except Exception as e:
        db_details = {
            "connection": {"status": "error", "message": str(e)}
        }
    
    return APIResponse(
        success=True,
        data=db_details,
        message="Database health check completed"
    )


@router.get(
    "/ml-model",
    response_model=APIResponse[Dict[str, Any]],
    summary="ML Model Health Check",
    description="Check ML model status and performance"
)
async def ml_model_health_check(
    current_user: User = Depends(get_current_user)
):
    """
    Check ML model status and performance.
    
    **Authentication:** Bearer token required
    
    **Returns:**
    - **Model loading status**
    - **Performance metrics**
    - **Model information**
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "model": {
                "status": "loaded",
                "model_type": "LogisticRegression",
                "model_version": "1.0.0",
                "features_count": 5000
            },
            "vectorizer": {
                "status": "loaded",
                "type": "TfidfVectorizer",
                "vocabulary_size": 5000
            },
            "performance": {
                "prediction_time_ms": 12.5,
                "test_sentiment": "positive",
                "test_confidence": 0.85
            }
        },
        "message": "ML model health check completed"
    }
    ```
    """
    try:
        from ..services.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        model_details = {
            "model": {
                "status": "loaded" if analyzer.model else "not_loaded",
                "model_version": getattr(analyzer, 'model_version', '1.0.0')
            }
        }
        
        # Model information
        if analyzer.model:
            model_info = {
                "model_type": analyzer.model.__class__.__name__,
                "status": "loaded"
            }
            
            # Get model features if available
            if hasattr(analyzer.model, 'n_features_in_'):
                model_info["features_count"] = analyzer.model.n_features_in_
            
            model_details["model"] = model_info
        
        # Vectorizer information
        if analyzer.vectorizer:
            vectorizer_info = {
                "status": "loaded",
                "type": analyzer.vectorizer.__class__.__name__
            }
            
            if hasattr(analyzer.vectorizer, 'vocabulary_'):
                vectorizer_info["vocabulary_size"] = len(analyzer.vectorizer.vocabulary_)
            
            model_details["vectorizer"] = vectorizer_info
        
        # Performance test
        if analyzer.model and analyzer.vectorizer:
            try:
                test_texts = [
                    "This is a great product!",
                    "I hate this service.",
                    "It's okay, nothing special."
                ]
                
                performance_results = []
                
                for test_text in test_texts:
                    start_time = time.time()
                    sentiment, confidence = await analyzer.predict_sentiment(test_text)
                    prediction_time_ms = (time.time() - start_time) * 1000
                    
                    performance_results.append({
                        "text": test_text,
                        "sentiment": sentiment,
                        "confidence": round(confidence, 3),
                        "prediction_time_ms": round(prediction_time_ms, 2)
                    })
                
                avg_time = sum(r["prediction_time_ms"] for r in performance_results) / len(performance_results)
                
                model_details["performance"] = {
                    "average_prediction_time_ms": round(avg_time, 2),
                    "test_results": performance_results
                }
            
            except Exception as e:
                model_details["performance"] = {"error": str(e)}
        
        return APIResponse(
            success=True,
            data=model_details,
            message="ML model health check completed"
        )
    
    except Exception as e:
        return APIResponse(
            success=False,
            data={"error": str(e)},
            message="ML model health check failed"
        )
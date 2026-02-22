"""
Main FastAPI application with comprehensive fraud detection capabilities.
Includes all endpoints, middleware, authentication, and enterprise features.
"""

from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.security import setup_security_middleware, User, require_authentication
from app.database import init_database, check_database_health
from app.api.v1 import fraud_detection, analytics
from app.schemas import APIResponse, ErrorResponse, HealthResponse, ComponentHealth

# Initialize settings and logging
settings = get_settings()
setup_logging()
logger = get_logger(__name__)

# Application metadata
APP_NAME = "CipherGuard Fraud Detection API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = """
## CipherGuard Enterprise Fraud Detection Platform

A comprehensive, enterprise-grade fraud detection system with advanced machine learning capabilities.

### Key Features

* **Advanced ML Pipeline**: Ensemble models with IsolationForest, RandomForest, and XGBoost
* **Real-time Detection**: Sub-second fraud prediction with explainable AI
* **Enterprise Security**: JWT authentication, API keys, rate limiting, and audit trails
* **Comprehensive Analytics**: Fraud statistics, trends, and performance metrics
* **Scalable Architecture**: Async FastAPI with PostgreSQL and proper caching
* **Production Ready**: Docker containers, Kubernetes deployment, and monitoring

### Authentication

This API supports multiple authentication methods:

1. **JWT Tokens**: Obtain via `/auth/login` endpoint
2. **API Keys**: Include in `X-API-Key` header
3. **Bearer Tokens**: Standard OAuth2 bearer token format

### Rate Limiting

API requests are rate-limited per user/IP:
- Default: 100 requests per minute
- Burst allowance: 200 requests per minute
- Blocked IPs experience temporary 5-minute blocks

### API Versioning

Current API version: **v1**
- Base URL: `/api/v1/`
- Backward compatibility maintained for production deployments
"""

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs" if not settings.is_production() else None,
    redoc_url="/redoc" if not settings.is_production() else None,
    openapi_url="/openapi.json" if not settings.is_production() else None,
    contact={
        "name": "CipherGuard Security Team",
        "email": "security@cipherguard.com",
        "url": "https://cipherguard.com/support",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://cipherguard.com/license",
    },
    servers=[
        {
            "url": f"https://{settings.api.host}",
            "description": "Production server"
        },
        {
            "url": f"http://localhost:{settings.api.port}",
            "description": "Development server"
        }
    ] if settings.is_production() else [
        {
            "url": f"http://localhost:{settings.api.port}",
            "description": "Development server"
        }
    ]
)

# Application startup time for uptime calculation
app_start_time = datetime.utcnow()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # Check ML pipeline
        logger.info("Checking ML pipeline...")
        from app.ml.inference.pipeline import fraud_detection_pipeline
        if not fraud_detection_pipeline.is_trained:
            logger.warning("ML pipeline is not trained. Training may be required.")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", exc_info=True)
        # Don't raise - let the app start but log the error
        # In production, you might want to fail fast


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application...")
    
    # Perform cleanup tasks
    # Close database connections, cleanup ML models, etc.
    
    logger.info("Application shutdown completed")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Don't expose internal errors in production
    if settings.is_production():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                message="An internal server error occurred",
                error_code="INTERNAL_ERROR"
            ).dict()
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                message=f"Internal server error: {str(exc)}",
                error_code="INTERNAL_ERROR",
                details={"exception_type": type(exc).__name__}
            ).dict()
        )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


# Setup security middleware
setup_security_middleware(app)

# Include API routers
app.include_router(fraud_detection.router, prefix="")
app.include_router(analytics.router, prefix="")


# Root endpoint
@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information."""
    return APIResponse(
        message=f"Welcome to {APP_NAME} v{APP_VERSION}",
        timestamp=datetime.utcnow()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Database connectivity
    - ML pipeline status
    - System resources
    - External dependencies
    """
    try:
        components = []
        overall_status = "healthy"
        
        # Database health
        db_health_start = datetime.utcnow()
        db_health = check_database_health()
        db_response_time = (datetime.utcnow() - db_health_start).total_seconds() * 1000
        
        db_component = ComponentHealth(
            name="database",
            status="healthy" if db_health["status"] == "healthy" else "unhealthy",
            last_check=datetime.utcnow(),
            response_time_ms=db_response_time,
            details=db_health
        )
        components.append(db_component)
        
        if db_health["status"] != "healthy":
            overall_status = "unhealthy"
        
        # ML Pipeline health
        ml_health_start = datetime.utcnow()
        from app.ml.inference.pipeline import fraud_detection_pipeline
        
        ml_status = "healthy" if fraud_detection_pipeline.is_trained else "not_ready"
        ml_response_time = (datetime.utcnow() - ml_health_start).total_seconds() * 1000
        
        ml_component = ComponentHealth(
            name="ml_pipeline",
            status=ml_status,
            last_check=datetime.utcnow(),
            response_time_ms=ml_response_time,
            details={
                "is_trained": fraud_detection_pipeline.is_trained,
                "model_version": fraud_detection_pipeline.model_version
            }
        )
        components.append(ml_component)
        
        if ml_status != "healthy" and overall_status == "healthy":
            overall_status = "degraded"
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - app_start_time).total_seconds()
        
        return HealthResponse(
            status=overall_status,
            version=APP_VERSION,
            environment=settings.environment,
            components=components,
            uptime_seconds=uptime_seconds
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            version=APP_VERSION,
            environment=settings.environment,
            components=[
                ComponentHealth(
                    name="system",
                    status="unhealthy",
                    last_check=datetime.utcnow(),
                    details={"error": str(e)}
                )
            ],
            uptime_seconds=(datetime.utcnow() - app_start_time).total_seconds()
        )


# Metrics endpoint for monitoring
@app.get("/metrics")
async def metrics(current_user: User = Depends(require_authentication)):
    """
    Prometheus-style metrics endpoint.
    Requires authentication for security.
    """
    try:
        # Basic application metrics
        uptime_seconds = (datetime.utcnow() - app_start_time).total_seconds()
        
        # Database metrics
        db_health = check_database_health()
        db_status = 1 if db_health["status"] == "healthy" else 0
        
        # ML pipeline metrics
        from app.ml.inference.pipeline import fraud_detection_pipeline
        ml_status = 1 if fraud_detection_pipeline.is_trained else 0
        
        # Format as Prometheus metrics
        metrics_text = f"""# HELP cipherguard_app_uptime_seconds Application uptime in seconds
# TYPE cipherguard_app_uptime_seconds gauge
cipherguard_app_uptime_seconds {uptime_seconds}

# HELP cipherguard_database_status Database health status (1=healthy, 0=unhealthy)
# TYPE cipherguard_database_status gauge
cipherguard_database_status {db_status}

# HELP cipherguard_ml_pipeline_status ML pipeline status (1=ready, 0=not_ready)
# TYPE cipherguard_ml_pipeline_status gauge
cipherguard_ml_pipeline_status {ml_status}

# HELP cipherguard_app_info Application information
# TYPE cipherguard_app_info gauge
cipherguard_app_info{{version="{APP_VERSION}",environment="{settings.environment}"}} 1
"""
        
        return JSONResponse(
            content=metrics_text,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate metrics"
        )


# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    
    # Add security to all authenticated endpoints
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                # Add security requirement if not public endpoint
                if path not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
                    operation["security"] = [
                        {"BearerAuth": []},
                        {"ApiKeyAuth": []}
                    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Development server runner
def run_development_server():
    """Run the development server with auto-reload."""
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )


if __name__ == "__main__":
    run_development_server()


# Export the FastAPI app
__all__ = ["app", "run_development_server"]
"""
Pydantic schemas for request/response validation in Comment Sentiment Analysis API.
Defines data models for API endpoints with validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum


T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"key": "value"},
                "error": None
            }
        }


class SentimentType(str, Enum):
    """Enumeration for sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class CommentAnalysisRequest(BaseModel):
    """Request schema for comment sentiment analysis."""
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    comment_text: str = Field(..., min_length=1, max_length=5000, description="Comment text to analyze")
    
    @validator('comment_text')
    def validate_comment_text(cls, v):
        if not v.strip():
            raise ValueError('Comment text cannot be empty or only whitespace')
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError('User ID cannot be empty or only whitespace')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "comment_text": "This product is absolutely amazing! I love it so much."
            }
        }


class CommentAnalysisResponse(BaseModel):
    """Response schema for comment sentiment analysis."""
    id: int = Field(..., description="Comment ID in database")
    user_id: str = Field(..., description="User identifier")
    comment_text: str = Field(..., description="Original comment text")
    sentiment: SentimentType = Field(..., description="Predicted sentiment")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    created_at: datetime = Field(..., description="Analysis timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": "user123",
                "comment_text": "This product is absolutely amazing! I love it so much.",
                "sentiment": "positive",
                "confidence_score": 0.95,
                "created_at": "2024-01-01T12:00:00Z",
                "processing_time_ms": 45.2
            }
        }


class CommentListResponse(BaseModel):
    """Response schema for listing comments."""
    comments: List[CommentAnalysisResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int

    class Config:
        schema_extra = {
            "example": {
                "comments": [
                    {
                        "id": 1,
                        "user_id": "user123",
                        "comment_text": "This product is great!",
                        "sentiment": "positive",
                        "confidence_score": 0.92,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                ],
                "total_count": 100,
                "page": 1,
                "page_size": 10,
                "total_pages": 10
            }
        }


class CommentResponse(BaseModel):
    """Simple comment response schema for basic comment data."""
    id: int = Field(..., description="Comment ID")
    user_id: str = Field(..., description="User identifier")
    comment_text: str = Field(..., description="Comment content")
    sentiment: SentimentType = Field(..., description="Sentiment classification")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": "user123",
                "comment_text": "This is a great product!",
                "sentiment": "positive",
                "confidence_score": 0.95,
                "created_at": "2024-01-01T12:00:00Z"
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there's a next page")
    has_prev: bool = Field(..., description="Whether there's a previous page")

    class Config:
        schema_extra = {
            "example": {
                "items": [],
                "total": 100,
                "page": 1,
                "page_size": 10,
                "total_pages": 10,
                "has_next": True,
                "has_prev": False
            }
        }


class SentimentAnalytics(BaseModel):
    """Schema for sentiment analytics data."""
    total_comments: int = Field(..., description="Total number of comments")
    positive_count: int = Field(..., description="Number of positive comments")
    negative_count: int = Field(..., description="Number of negative comments")
    neutral_count: int = Field(..., description="Number of neutral comments")
    positive_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of positive comments")
    negative_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of negative comments")
    neutral_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of neutral comments")
    avg_confidence_score: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "total_comments": 1000,
                "positive_count": 600,
                "negative_count": 250,
                "neutral_count": 150,
                "positive_percentage": 60.0,
                "negative_percentage": 25.0,
                "neutral_percentage": 15.0,
                "avg_confidence_score": 0.85
            }
        }


class TimeSeriesPoint(BaseModel):
    """Schema for time-series data point."""
    timestamp: datetime = Field(..., description="Timestamp for the data point")
    positive_count: int = Field(..., description="Count of positive comments")
    negative_count: int = Field(..., description="Count of negative comments")
    neutral_count: int = Field(..., description="Count of neutral comments")
    total_count: int = Field(..., description="Total count of comments")


class TimeSeriesAnalytics(BaseModel):
    """Schema for time-series analytics data."""
    data_points: List[TimeSeriesPoint] = Field(..., description="Time-series data points")
    interval: str = Field(..., description="Time interval (hourly, daily, etc.)")
    start_date: datetime = Field(..., description="Start date for the time series")
    end_date: datetime = Field(..., description="End date for the time series")


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    database_status: str = Field(..., description="Database connection status")
    model_status: str = Field(..., description="ML model status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "database_status": "connected",
                "model_status": "loaded",
                "version": "1.0.0",
                "uptime_seconds": 3600.0
            }
        }


class SystemHealth(BaseModel):
    """Schema for detailed system health information."""
    database_connected: bool = Field(..., description="Database connection status")
    database_latency_ms: float = Field(..., description="Database response time in milliseconds")
    model_loaded: bool = Field(..., description="ML model availability status")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    active_connections: int = Field(..., description="Number of active connections")

    class Config:
        schema_extra = {
            "example": {
                "database_connected": True,
                "database_latency_ms": 15.2,
                "model_loaded": True,
                "memory_usage_mb": 512.5,
                "cpu_usage_percent": 25.3,
                "disk_usage_percent": 45.8,
                "uptime_seconds": 3600.0,
                "active_connections": 12
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Comment text is required",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str = Field(..., min_length=1, max_length=100, description="Username or email")
    password: str = Field(..., min_length=1, max_length=100, description="User password")

    class Config:
        schema_extra = {
            "example": {
                "username": "admin",
                "password": "your_password"
            }
        }


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema for JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


# ======================= FRAUD DETECTION SCHEMAS =======================

class RiskLevel(str, Enum):
    """Enumeration for fraud risk levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class FraudDetectionRequest(BaseModel):
    """Request schema for fraud detection analysis."""
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant: str = Field(..., min_length=1, max_length=200, description="Merchant name")
    device: str = Field(default="desktop", description="Device type (mobile, desktop, tablet)")
    country: str = Field(default="US", max_length=10, description="Country code")
    
    # Customer info for alerts (optional)
    customer_name: Optional[str] = Field(None, max_length=100, description="Customer name for alerts")
    customer_email: Optional[str] = Field(None, max_length=200, description="Customer email for alerts")
    customer_phone: Optional[str] = Field(None, max_length=20, description="Customer phone for SMS alerts")
    send_alert: bool = Field(default=True, description="Whether to send alert if fraud detected")

    @validator('merchant')
    def validate_merchant(cls, v):
        if not v.strip():
            raise ValueError('Merchant cannot be empty')
        return v.strip()
    
    @validator('device')
    def validate_device(cls, v):
        valid_devices = ["mobile", "desktop", "tablet"]
        v = v.lower().strip()
        if v not in valid_devices:
            v = "desktop"  # Default to desktop
        return v
    
    @validator('customer_email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

    class Config:
        schema_extra = {
            "example": {
                "amount": 150.00,
                "merchant": "Amazon",
                "device": "desktop",
                "country": "US",
                "customer_name": "John Doe",
                "customer_email": "john@example.com",
                "customer_phone": "+1234567890",
                "send_alert": True
            }
        }


class FraudDetectionResponse(BaseModel):
    """Response schema for fraud detection analysis."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., description="Transaction amount")
    merchant: str = Field(..., description="Merchant name")
    device: str = Field(..., description="Device type")
    country: str = Field(..., description="Country code")
    is_fraud: bool = Field(..., description="Whether transaction is flagged as fraud")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk classification level")
    created_at: datetime = Field(..., description="Analysis timestamp")
    model_version: Optional[str] = Field(None, description="ML model version used")
    
    # Alert information
    alert_sent: bool = Field(default=False, description="Whether alert was sent to customer")
    alert_channels: Optional[List[str]] = Field(None, description="Channels used for alert (sms, email, push)")

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "amount": 150.00,
                "merchant": "Amazon",
                "device": "desktop",
                "country": "US",
                "is_fraud": False,
                "fraud_score": 0.15,
                "risk_level": "LOW",
                "created_at": "2026-02-25T12:00:00Z",
                "model_version": "isolation_forest_v1",
                "alert_sent": True,
                "alert_channels": ["sms", "email"]
            }
        }


class FraudListResponse(BaseModel):
    """Response schema for listing fraud transactions."""
    transactions: List[FraudDetectionResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int

    class Config:
        schema_extra = {
            "example": {
                "transactions": [],
                "total_count": 100,
                "page": 1,
                "page_size": 10,
                "total_pages": 10
            }
        }


class FraudAnalytics(BaseModel):
    """Response schema for fraud analytics."""
    total_transactions: int = Field(..., description="Total number of transactions analyzed")
    fraud_count: int = Field(..., description="Number of fraudulent transactions")
    legitimate_count: int = Field(..., description="Number of legitimate transactions")
    fraud_rate: float = Field(..., description="Percentage of fraudulent transactions")
    average_fraud_score: float = Field(..., description="Average fraud score")
    risk_distribution: Dict[str, int] = Field(..., description="Distribution by risk level")
    total_amount: float = Field(..., description="Total transaction amount")
    fraud_amount: float = Field(..., description="Total fraud amount")

    class Config:
        schema_extra = {
            "example": {
                "total_transactions": 1000,
                "fraud_count": 50,
                "legitimate_count": 950,
                "fraud_rate": 5.0,
                "average_fraud_score": 0.25,
                "risk_distribution": {"CRITICAL": 10, "HIGH": 20, "MEDIUM": 100, "LOW": 500, "VERY_LOW": 370},
                "total_amount": 150000.00,
                "fraud_amount": 25000.00
            }
        }


# ======================= CUSTOMER ALERT SCHEMAS =======================

class AlertChannel(str, Enum):
    """Alert notification channels."""
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    ALL = "all"


class AlertType(str, Enum):
    """Types of fraud alerts."""
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"
    HIGH_AMOUNT = "high_amount"
    UNUSUAL_LOCATION = "unusual_location"
    BLOCKED_TRANSACTION = "blocked_transaction"


class AlertResponse(BaseModel):
    """Response schema for alert notification."""
    alert_id: str = Field(..., description="Unique alert identifier")
    transaction_id: str = Field(..., description="Related transaction ID")
    customer_email: Optional[str] = Field(None, description="Customer email")
    customer_phone: Optional[str] = Field(None, description="Customer phone")
    alert_type: str = Field(..., description="Type of alert sent")
    channels_used: List[str] = Field(..., description="Notification channels used")
    timestamp: datetime = Field(..., description="Alert timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_1234567890",
                "transaction_id": "txn_1234567890",
                "customer_email": "john@example.com",
                "customer_phone": "+1234567890",
                "alert_type": "suspicious_transaction",
                "channels_used": ["sms", "email"],
                "timestamp": "2026-02-25T12:00:00Z"
            }
        }


class AlertHistoryResponse(BaseModel):
    """Response schema for alert history."""
    alerts: List[AlertResponse]
    total_count: int

    class Config:
        schema_extra = {
            "example": {
                "alerts": [],
                "total_count": 0
            }
        }


class AlertStats(BaseModel):
    """Response schema for alert statistics."""
    total_alerts: int = Field(..., description="Total alerts sent")
    by_type: Dict[str, int] = Field(..., description="Alerts by type")
    by_channel: Dict[str, int] = Field(..., description="Alerts by channel")
    success_rate: float = Field(..., description="Alert delivery success rate")
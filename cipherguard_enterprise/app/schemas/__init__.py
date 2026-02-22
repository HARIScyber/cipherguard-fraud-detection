"""
Pydantic schemas for API request/response validation.
Defines data models for all API endpoints.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import UUID

from ..models import RiskLevel, FeedbackType, TransactionStatus


class APIResponse(BaseModel):
    """Base API response model."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class ErrorResponse(APIResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Transaction schemas
class TransactionBase(BaseModel):
    """Base transaction schema."""
    user_id: UUID
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    transaction_type: str = Field(..., min_length=1, max_length=50)
    merchant_name: Optional[str] = Field(None, max_length=255)
    merchant_category: Optional[str] = Field(None, max_length=100)
    
    # Location and device data
    ip_address: Optional[str] = Field(None, max_length=45)
    country_code: Optional[str] = Field(None, min_length=2, max_length=2)
    city: Optional[str] = Field(None, max_length=100)
    device_fingerprint: Optional[str] = Field(None, max_length=255)
    user_agent: Optional[str] = None
    
    # Timing data
    transaction_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    local_time: Optional[datetime] = None
    timezone_offset: Optional[int] = Field(None, ge=-720, le=840)  # Minutes from UTC
    
    # Channel and context
    channel: Optional[str] = Field(None, max_length=50)
    session_id: Optional[str] = Field(None, max_length=255)
    reference_number: Optional[str] = Field(None, max_length=100)
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:  # $1M limit for safety
            raise ValueError('Amount exceeds maximum allowed limit')
        return v
    
    @validator('country_code')
    def validate_country_code(cls, v):
        if v and len(v) != 2:
            raise ValueError('Country code must be exactly 2 characters')
        return v.upper() if v else v


class TransactionCreate(TransactionBase):
    """Schema for creating a new transaction."""
    pass


class TransactionResponse(TransactionBase):
    """Schema for transaction response."""
    id: UUID
    status: TransactionStatus
    processing_time_ms: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Fraud detection schemas
class FraudDetectionRequest(BaseModel):
    """Request schema for fraud detection."""
    transaction: TransactionCreate
    return_explanation: bool = Field(default=True, description="Include SHAP explanations")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")


class RiskFactor(BaseModel):
    """Risk factor in fraud explanation."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance score")
    value: Optional[Any] = Field(None, description="Feature value for this transaction")


class FraudDetectionResponse(APIResponse):
    """Response schema for fraud detection."""
    transaction_id: UUID
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence")
    
    # Model information
    model_name: str = Field(default="ensemble", description="Model used for prediction")
    model_version: Optional[str] = Field(None, description="Model version")
    prediction_latency_ms: float = Field(..., description="Prediction processing time")
    
    # Thresholds used
    fraud_threshold: float = Field(..., description="Threshold used for fraud classification")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="All thresholds used")
    
    # Explanations (optional)
    top_risk_factors: Optional[List[RiskFactor]] = Field(None, description="Top contributing risk factors")
    feature_importance: Optional[List[float]] = Field(None, description="Complete feature importance array")
    shap_values: Optional[List[float]] = Field(None, description="SHAP values for explainability")


# Feedback schemas
class FeedbackCreate(BaseModel):
    """Schema for creating feedback."""
    transaction_id: UUID
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    is_fraud_actual: bool = Field(..., description="Actual fraud status")
    confidence: Optional[str] = Field(None, description="Confidence level (high/medium/low)")
    notes: Optional[str] = Field(None, description="Additional notes or context")
    evidence_provided: Optional[Dict[str, Any]] = Field(None, description="Supporting evidence")


class FeedbackResponse(APIResponse):
    """Response schema for feedback."""
    feedback_id: UUID
    transaction_id: UUID
    feedback_type: FeedbackType
    is_fraud_actual: bool
    review_status: str = Field(default="pending")
    created_at: datetime
    
    class Config:
        from_attributes = True


# Analytics schemas
class TimeRange(BaseModel):
    """Time range for analytics queries."""
    start_time: datetime
    end_time: datetime
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class FraudStatistics(BaseModel):
    """Fraud statistics response."""
    time_range: TimeRange
    total_transactions: int = Field(..., ge=0)
    fraud_transactions: int = Field(..., ge=0)
    fraud_rate: float = Field(..., ge=0, le=100, description="Fraud rate as percentage")
    total_amount: float = Field(..., ge=0)
    fraud_amount: float = Field(..., ge=0)
    
    # Risk level distribution
    risk_level_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Top merchants/countries with fraud
    top_fraud_merchants: List[Dict[str, Any]] = Field(default_factory=list)
    top_fraud_countries: List[Dict[str, Any]] = Field(default_factory=list)


class ModelPerformance(BaseModel):
    """Model performance metrics."""
    model_name: str
    model_version: str
    time_range: TimeRange
    
    # Performance metrics
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    roc_auc: float = Field(..., ge=0, le=1)
    
    # Confusion matrix
    true_positives: int = Field(..., ge=0)
    true_negatives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    
    # Volume and latency
    total_predictions: int = Field(..., ge=0)
    avg_prediction_latency_ms: float = Field(..., ge=0)


class AnalyticsResponse(APIResponse):
    """Comprehensive analytics response."""
    fraud_statistics: FraudStatistics
    model_performance: List[ModelPerformance]
    system_health: Dict[str, Any] = Field(default_factory=dict)


# Model management schemas
class ModelInfo(BaseModel):
    """Model information schema."""
    model_name: str
    model_version: str
    model_type: str = Field(..., description="Type of model (ensemble, xgboost, etc.)")
    is_active: bool = Field(default=False)
    training_date: datetime
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    feature_count: int = Field(..., ge=0)


class ModelsResponse(APIResponse):
    """Response schema for model listing."""
    models: List[ModelInfo]
    active_model: Optional[ModelInfo] = None


class ModelTrainingRequest(BaseModel):
    """Request schema for model training."""
    training_data_source: str = Field(..., description="Source of training data")
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration overrides")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation data split ratio")
    hyperparameter_tuning: bool = Field(default=True, description="Enable hyperparameter tuning")


class ModelTrainingResponse(APIResponse):
    """Response schema for model training."""
    training_job_id: UUID
    model_version: str
    status: str = Field(default="started")
    estimated_completion_time: Optional[datetime] = None


# Health check schemas
class ComponentHealth(BaseModel):
    """Individual component health status."""
    name: str
    status: str = Field(..., description="healthy/unhealthy/degraded")
    last_check: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    environment: str
    components: List[ComponentHealth]
    uptime_seconds: float = Field(..., ge=0)


# User management schemas
class UserCreate(BaseModel):
    """Schema for creating a user."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="Valid email address")
    full_name: Optional[str] = Field(None, max_length=255)
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    
    @validator('email')
    def validate_email(cls, v):
        # Basic email validation
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower()


class UserResponse(BaseModel):
    """User response schema."""
    id: UUID
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    per_page: int = Field(..., ge=1)
    pages: int = Field(..., ge=1)
    has_next: bool
    has_prev: bool


# Export all schemas
__all__ = [
    'APIResponse',
    'ErrorResponse',
    'TransactionCreate',
    'TransactionResponse',
    'FraudDetectionRequest',
    'FraudDetectionResponse',
    'RiskFactor',
    'FeedbackCreate',
    'FeedbackResponse',
    'TimeRange',
    'FraudStatistics',
    'ModelPerformance', 
    'AnalyticsResponse',
    'ModelInfo',
    'ModelsResponse',
    'ModelTrainingRequest',
    'ModelTrainingResponse',
    'ComponentHealth',
    'HealthResponse',
    'UserCreate',
    'UserResponse',
    'TokenResponse',
    'PaginationParams',
    'PaginatedResponse'
]
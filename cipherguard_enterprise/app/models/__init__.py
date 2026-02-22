"""
Database models for fraud detection system.
Includes Users, Transactions, Predictions, Feedback, and Audit tables.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from enum import Enum as PyEnum

from ..database import Base


class RiskLevel(PyEnum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class TransactionStatus(PyEnum):
    """Transaction status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    FLAGGED = "flagged"


class FeedbackType(PyEnum):
    """Feedback type enumeration."""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"


class User(Base):
    """User model for authentication and tracking."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True))
    
    # API key for programmatic access
    api_key = Column(String(255), unique=True, index=True)
    api_key_created_at = Column(DateTime(timezone=True))
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Transaction(Base):
    """Transaction model with comprehensive fraud detection features."""
    
    __tablename__ = "transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Basic transaction data
    amount = Column(Float, nullable=False, index=True)
    currency = Column(String(3), nullable=False, default="USD")
    transaction_type = Column(String(50), nullable=False, index=True)  # purchase, withdrawal, transfer, etc.
    merchant_name = Column(String(255), index=True)
    merchant_category = Column(String(100), index=True)
    
    # Location and device data
    ip_address = Column(String(45))  # IPv6 compatible
    country_code = Column(String(2), index=True)
    city = Column(String(100))
    device_fingerprint = Column(String(255), index=True)
    user_agent = Column(Text)
    
    # Timing data
    transaction_time = Column(DateTime(timezone=True), nullable=False, index=True)
    local_time = Column(DateTime(timezone=True))
    timezone_offset = Column(Integer)  # Minutes from UTC
    
    # Status and processing
    status = Column(String(20), nullable=False, default=TransactionStatus.PENDING.value, index=True)
    processing_time_ms = Column(Integer)
    
    # Additional metadata
    channel = Column(String(50), index=True)  # web, mobile, api, etc.
    session_id = Column(String(255), index=True)
    reference_number = Column(String(100), unique=True, index=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    predictions = relationship("Prediction", back_populates="transaction")
    feedback = relationship("Feedback", back_populates="transaction")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_transaction_user_time", "user_id", "transaction_time"),
        Index("idx_transaction_amount_time", "amount", "transaction_time"),
        Index("idx_transaction_merchant_time", "merchant_name", "transaction_time"),
        Index("idx_transaction_ip_time", "ip_address", "transaction_time"),
    )
    
    def __repr__(self):
        return f"<Transaction(id={self.id}, amount={self.amount}, user_id={self.user_id})>"


class Prediction(Base):
    """ML prediction results for transactions."""
    
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=False, index=True)
    
    # Prediction results
    fraud_probability = Column(Float, nullable=False, index=True)  # 0.0 to 1.0
    risk_level = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    is_fraud = Column(Boolean, nullable=False, index=True)
    
    # Model information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    ensemble_models = Column(JSON)  # List of models used in ensemble
    
    # Feature importance and explainability
    feature_importance = Column(JSON)  # Feature names and importance scores
    shap_values = Column(JSON)  # SHAP values for explainability
    top_risk_factors = Column(JSON)  # Top factors contributing to fraud score
    
    # Prediction confidence and metadata
    confidence_score = Column(Float)  # Model confidence in prediction
    prediction_latency_ms = Column(Integer)  # Time taken for prediction
    
    # Thresholds used
    fraud_threshold = Column(Float, nullable=False)
    high_risk_threshold = Column(Float)
    medium_risk_threshold = Column(Float)
    
    # Processing metadata
    correlation_id = Column(String(100), index=True)  # For request tracing
    api_version = Column(String(10))
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    transaction = relationship("Transaction", back_populates="predictions")
    
    # Indexes for analytics
    __table_args__ = (
        Index("idx_prediction_fraud_prob", "fraud_probability"),
        Index("idx_prediction_model_time", "model_name", "created_at"),
        Index("idx_prediction_risk_time", "risk_level", "created_at"),
    )
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, fraud_prob={self.fraud_probability}, transaction_id={self.transaction_id})>"


class Feedback(Base):
    """Human feedback on fraud predictions for model improvement."""
    
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Feedback data
    feedback_type = Column(String(20), nullable=False, index=True)  # TP, FP, TN, FN
    is_fraud_actual = Column(Boolean, nullable=False, index=True)
    confidence = Column(String(20))  # high, medium, low
    
    # Additional context
    notes = Column(Text)
    evidence_provided = Column(JSON)  # Supporting evidence or documentation
    investigation_outcome = Column(Text)
    
    # Reviewer information
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    review_status = Column(String(20), default="pending", index=True)  # pending, approved, rejected
    reviewed_at = Column(DateTime(timezone=True))
    
    # Impact tracking
    model_retrained = Column(Boolean, default=False)
    retrain_date = Column(DateTime(timezone=True))
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    transaction = relationship("Transaction", back_populates="feedback")
    user = relationship("User", back_populates="feedback", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewer_id])
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, type={self.feedback_type}, transaction_id={self.transaction_id})>"


class ModelMetrics(Base):
    """Model performance metrics over time."""
    
    __tablename__ = "model_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Model information
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    
    # Time period for metrics
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    auc_pr = Column(Float)
    
    # Confusion matrix
    true_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    # Volume metrics
    total_predictions = Column(Integer, default=0, index=True)
    fraud_predictions = Column(Integer, default=0)
    
    # Performance metrics
    avg_prediction_latency_ms = Column(Float)
    p95_prediction_latency_ms = Column(Float)
    p99_prediction_latency_ms = Column(Float)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name}, accuracy={self.accuracy})>"


class AuditLog(Base):
    """Audit log for security and compliance tracking."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event information
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)  # auth, prediction, admin, etc.
    severity = Column(String(20), nullable=False, index=True)  # info, warning, error, critical
    
    # User and session context
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    session_id = Column(String(255))
    correlation_id = Column(String(100), index=True)
    
    # Event details
    description = Column(Text, nullable=False)
    additional_data = Column(JSON)
    
    # Request context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    endpoint = Column(String(255))
    http_method = Column(String(10))
    
    # Outcome
    success = Column(Boolean, index=True)
    error_message = Column(Text)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_audit_user_time", "user_id", "created_at"),
        Index("idx_audit_event_time", "event_type", "created_at"),
        Index("idx_audit_severity_time", "severity", "created_at"),
    )
    
    def __repr__(self):
        return f"<AuditLog(event_type={self.event_type}, user_id={self.user_id})>"


# Export all models
__all__ = [
    'User',
    'Transaction', 
    'Prediction',
    'Feedback',
    'ModelMetrics',
    'AuditLog',
    'RiskLevel',
    'TransactionStatus',
    'FeedbackType'
]
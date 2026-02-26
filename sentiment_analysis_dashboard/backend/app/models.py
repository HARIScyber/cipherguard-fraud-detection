"""
SQLAlchemy models for Comment Sentiment Analysis Dashboard.
Defines database schema for comments, users, and analytics.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.sql import func
from datetime import datetime
from .database import Base


class Comment(Base):
    """
    Comment model to store analyzed comments with sentiment predictions.
    """
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    comment_text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False, index=True)  # positive, negative, neutral
    confidence_score = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional metadata
    text_length = Column(Integer)
    processing_time_ms = Column(Float)
    model_version = Column(String(50), default="1.0")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_user_sentiment", "user_id", "sentiment"),
        Index("idx_created_sentiment", "created_at", "sentiment"),
        Index("idx_confidence_sentiment", "confidence_score", "sentiment"),
    )
    
    def __repr__(self):
        return f"<Comment(id={self.id}, sentiment='{self.sentiment}', confidence={self.confidence_score:.2f})>"


class User(Base):
    """
    User model for authentication and user management.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    
    # User status and permissions
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class AnalyticsCache(Base):
    """
    Cache table for pre-computed analytics to improve dashboard performance.
    """
    __tablename__ = "analytics_cache"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cache_key = Column(String(100), unique=True, nullable=False, index=True)
    cache_value = Column(Text, nullable=False)  # JSON string
    
    # Cache metadata
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<AnalyticsCache(key='{self.cache_key}', expires_at={self.expires_at})>"


class SystemMetrics(Base):
    """
    System metrics for monitoring API performance and usage.
    """
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    
    # Metric metadata
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    tags = Column(Text)  # JSON string for additional tags
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Index for time-series queries
    __table_args__ = (
        Index("idx_metric_time", "metric_name", "recorded_at"),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"


class FraudTransaction(Base):
    """
    Fraud transaction model to store analyzed transactions with fraud predictions.
    """
    __tablename__ = "fraud_transactions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Transaction details
    amount = Column(Float, nullable=False)
    merchant = Column(String(200), nullable=False)
    device = Column(String(50), nullable=False)
    country = Column(String(10), nullable=False)
    
    # Customer info (for alerts)
    customer_name = Column(String(100), nullable=True)
    customer_email = Column(String(200), nullable=True)
    customer_phone = Column(String(20), nullable=True)
    
    # Fraud detection results
    is_fraud = Column(Boolean, nullable=False, index=True)
    fraud_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False, index=True)  # CRITICAL, HIGH, MEDIUM, LOW, VERY_LOW
    
    # Alert status
    alert_sent = Column(Boolean, default=False)
    alert_channels = Column(String(100), nullable=True)  # Comma-separated: sms,email,push
    
    # Model information
    model_version = Column(String(50), default="isolation_forest_v1")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_fraud_risk", "is_fraud", "risk_level"),
        Index("idx_fraud_created", "created_at", "is_fraud"),
        Index("idx_fraud_amount", "amount", "is_fraud"),
    )
    
    def __repr__(self):
        return f"<FraudTransaction(id={self.transaction_id}, is_fraud={self.is_fraud}, risk_level='{self.risk_level}')>"


class CustomerAlert(Base):
    """
    Customer alert model to store notification history.
    """
    __tablename__ = "customer_alerts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    
    # Customer info
    customer_name = Column(String(100), nullable=True)
    customer_email = Column(String(200), nullable=True)
    customer_phone = Column(String(20), nullable=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # suspicious_transaction, high_amount, etc.
    channels_used = Column(String(100), nullable=False)  # sms,email,push
    
    # Message content (for audit)
    sms_message = Column(Text, nullable=True)
    email_subject = Column(String(200), nullable=True)
    
    # Delivery status
    sms_status = Column(String(20), default="pending")  # sent, failed, skipped
    email_status = Column(String(20), default="pending")
    push_status = Column(String(20), default="pending")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_alert_transaction", "transaction_id"),
        Index("idx_alert_created", "created_at"),
        Index("idx_alert_type", "alert_type"),
    )
    
    def __repr__(self):
        return f"<CustomerAlert(id={self.alert_id}, type='{self.alert_type}', transaction={self.transaction_id})>"
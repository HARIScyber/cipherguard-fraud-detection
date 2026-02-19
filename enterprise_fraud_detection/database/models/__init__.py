"""
Database models for the Enterprise Fraud Detection System
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, Numeric, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()


class Transaction(Base):
    """Transaction model to store all transaction data."""
    
    __tablename__ = "transactions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # Transaction details 
    amount = Column(Numeric(15, 2), nullable=False)
    merchant = Column(String(200), nullable=False, index=True)
    device_type = Column(String(50), nullable=False)
    country_code = Column(String(2), nullable=False, index=True)
    
    # User information
    customer_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Timestamps
    transaction_time = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Metadata
    raw_data = Column(Text, nullable=True)  # Store original JSON
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_customer_time', 'customer_id', 'transaction_time'),
        Index('idx_merchant_time', 'merchant', 'transaction_time'),
        Index('idx_amount_time', 'amount', 'transaction_time'),
    )


class FraudPrediction(Base):
    """Model to store fraud detection results."""
    
    __tablename__ = "fraud_predictions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    
    # Prediction results
    fraud_score = Column(Float, nullable=False, index=True)
    is_fraud = Column(Boolean, nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    confidence_score = Column(Float, nullable=False)
    
    # Model information
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # ensemble, isolation_forest, etc.
    
    # Individual model scores (for ensemble)
    isolation_forest_score = Column(Float, nullable=True)
    random_forest_score = Column(Float, nullable=True)
    xgboost_score = Column(Float, nullable=True)
    
    # Feature importance and explanations
    top_risk_factors = Column(Text, nullable=True)  # JSON array
    shap_values = Column(Text, nullable=True)  # JSON array
    
    # Processing metadata
    processing_time_ms = Column(Float, nullable=False)
    feature_vector = Column(Text, nullable=True)  # JSON array
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    # Indexes for analytics
    __table_args__ = (
        Index('idx_fraud_score_time', 'fraud_score', 'created_at'),
        Index('idx_risk_level_time', 'risk_level', 'created_at'),
    )


class UserBehavior(Base):
    """Model to track user behavior patterns."""
    
    __tablename__ = "user_behavior"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(100), nullable=False, index=True)
    
    # Behavior metrics (calculated over time windows)
    total_transactions = Column(Integer, default=0)
    total_amount = Column(Numeric(15, 2), default=0)
    avg_transaction_amount = Column(Numeric(15, 2), default=0)
    
    # Velocity metrics
    transactions_last_hour = Column(Integer, default=0)
    transactions_last_day = Column(Integer, default=0)
    amount_last_hour = Column(Numeric(15, 2), default=0)
    amount_last_day = Column(Numeric(15, 2), default=0)
    
    # Pattern metrics
    unique_merchants = Column(Integer, default=0)
    unique_countries = Column(Integer, default=0)
    unique_devices = Column(Integer, default=0)
    
    # Risk indicators
    failed_transactions = Column(Integer, default=0)
    disputed_transactions = Column(Integer, default=0)
    fraud_score_avg = Column(Float, default=0.0)
    
    # Timestamps
    first_transaction = Column(DateTime, nullable=True)
    last_transaction = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class UserFeedback(Base):
    """Model to store analyst feedback for model improvement."""
    
    __tablename__ = "user_feedback"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    
    # Feedback details
    is_actual_fraud = Column(Boolean, nullable=False)
    feedback_type = Column(String(50), nullable=False)  # manual_review, customer_dispute, etc.
    analyst_id = Column(String(100), nullable=True)
    
    # Additional context
    comments = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)  # Analyst confidence 0-1
    
    # Original prediction for comparison
    original_fraud_score = Column(Float, nullable=True)
    original_prediction = Column(Boolean, nullable=True)
    
    # Timestamps
    feedback_date = Column(DateTime, default=func.now(), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class ModelPerformance(Base):
    """Model to track ML model performance metrics."""
    
    __tablename__ = "model_performance"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Model information
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    training_date = Column(DateTime, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    
    # Training metadata
    training_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)
    training_time_seconds = Column(Float, nullable=False)
    
    # Cross-validation results
    cv_mean_accuracy = Column(Float, nullable=True)
    cv_std_accuracy = Column(Float, nullable=True)
    
    # Hyperparameters (stored as JSON)
    hyperparameters = Column(Text, nullable=True)
    
    # Feature importance
    feature_importance = Column(Text, nullable=True)  # JSON
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)


class SystemHealth(Base):
    """Model to track system health and performance."""
    
    __tablename__ = "system_health"
    
    # Primary key  
    id = Column(Integer, primary_key=True, index=True)
    
    # Metrics
    total_requests = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)
    failed_predictions = Column(Integer, default=0)
    avg_response_time_ms = Column(Float, default=0.0)
    
    # Fraud statistics
    total_fraud_detected = Column(Integer, default=0)
    fraud_rate_percent = Column(Float, default=0.0)
    
    # System resources
    cpu_usage_percent = Column(Float, default=0.0)
    memory_usage_mb = Column(Float, default=0.0)
    disk_usage_percent = Column(Float, default=0.0)
    
    # Model status
    active_model_version = Column(String(50), nullable=True)
    model_last_trained = Column(DateTime, nullable=True)
    
    # Time window for these metrics
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)


# Utility functions for database operations

def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all tables from the database."""
    Base.metadata.drop_all(bind=engine)
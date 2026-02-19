"""
Production-grade configuration management
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from decimal import Decimal

class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(..., env="DB_USERNAME")
    password: str = Field(..., env="DB_PASSWORD")
    database: str = Field(..., env="DB_DATABASE")
    
    # Connection pool settings
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    @property 
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisConfig(BaseSettings):
    """Redis configuration."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    # Connection pool settings
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: float = Field(default=5.0, env="REDIS_SOCKET_TIMEOUT")
    
class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_expiry_hours: int = Field(default=24, env="JWT_EXPIRY_HOURS")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # Encryption
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")
    
    @validator('jwt_secret')
    def jwt_secret_length(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return v
    
    @validator('encryption_key')
    def encryption_key_length(cls, v):
        if len(v) != 32:
            raise ValueError("Encryption key must be exactly 32 characters")
        return v

class FraudConfig(BaseSettings):
    """Fraud detection configuration."""
    
    # Thresholds
    fraud_threshold: float = Field(default=0.6, env="FRAUD_THRESHOLD")
    high_risk_threshold: float = Field(default=0.8, env="HIGH_RISK_THRESHOLD")
    
    # Transaction limits
    max_transaction_amount: Decimal = Field(default=Decimal("100000"), env="MAX_TRANSACTION_AMOUNT")
    velocity_window_minutes: int = Field(default=10, env="VELOCITY_WINDOW_MINUTES")
    max_transactions_per_window: int = Field(default=5, env="MAX_TRANSACTIONS_PER_WINDOW")
    
    # Model settings
    model_retrain_interval_hours: int = Field(default=24, env="MODEL_RETRAIN_INTERVAL")
    feature_vector_dimensions: int = Field(default=6, env="FEATURE_VECTOR_DIM")
    
    @validator('fraud_threshold', 'high_risk_threshold')
    def threshold_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Thresholds must be between 0 and 1")
        return v

class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Prometheus metrics
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    audit_log_file: str = Field(default="logs/audit.log", env="AUDIT_LOG_FILE")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    @validator('log_level')
    def valid_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8001, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # External services
    cyborg_api_key: Optional[str] = Field(None, env="CYBORGDB_API_KEY")
    
    # Feature flags
    enable_advanced_ml: bool = Field(default=False, env="ENABLE_ADVANCED_ML")
    enable_real_time_training: bool = Field(default=False, env="ENABLE_REAL_TIME_TRAINING")
    
    # Component configs
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig = SecurityConfig()
    fraud: FraudConfig = FraudConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
    
    @validator('environment')
    def valid_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v

# Global config instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

# Environment-specific overrides
def setup_production_config():
    """Apply production-specific configuration overrides."""
    if config.environment == "production":
        # Security hardening
        config.debug = False
        config.fraud.fraud_threshold = 0.5  # More sensitive in production
        config.security.rate_limit_per_minute = 30  # Stricter rate limiting
        
        # Performance optimization
        config.database.pool_size = 20
        config.redis.max_connections = 100
        
def setup_development_config():
    """Apply development-specific configuration overrides.""" 
    if config.environment == "development":
        config.debug = True
        config.monitoring.log_level = "DEBUG"
        config.fraud.fraud_threshold = 0.7  # Less sensitive for testing
"""
Core configuration settings for the Enterprise Fraud Detection System
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from decimal import Decimal


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://fraud_user:fraud_pass@localhost:5432/fraud_detection",
        env="DATABASE_URL"
    )
    echo: bool = Field(default=False, env="DB_ECHO")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE") 
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")


class MLSettings(BaseSettings):
    """Machine Learning configuration settings."""
    
    # Model settings
    models_dir: str = Field(default="ml/models", env="MODELS_DIR")
    ensemble_weights: List[float] = Field(default=[0.4, 0.3, 0.3])  # IsolationForest, RandomForest, XGBoost
    
    # Feature engineering
    feature_window_days: int = Field(default=30, env="FEATURE_WINDOW_DAYS")
    velocity_window_minutes: int = Field(default=60, env="VELOCITY_WINDOW_MINUTES")
    
    # Fraud thresholds
    low_risk_threshold: float = Field(default=0.3, env="LOW_RISK_THRESHOLD")
    medium_risk_threshold: float = Field(default=0.6, env="MEDIUM_RISK_THRESHOLD") 
    high_risk_threshold: float = Field(default=0.8, env="HIGH_RISK_THRESHOLD")
    
    # Model performance
    min_model_accuracy: float = Field(default=0.85, env="MIN_MODEL_ACCURACY")
    retrain_threshold: float = Field(default=0.80, env="RETRAIN_THRESHOLD")
    
    # Training settings
    test_size: float = Field(default=0.2, env="TEST_SIZE")
    random_state: int = Field(default=42, env="RANDOM_STATE")
    cv_folds: int = Field(default=5, env="CV_FOLDS")
    
    @validator('ensemble_weights')
    def weights_sum_to_one(cls, v):
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError('Ensemble weights must sum to 1.0')
        return v


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    @validator('secret_key')
    def secret_key_length(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: str = Field(default="logs/app.log", env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    @validator('level')
    def valid_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Metrics
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Alerting
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    alert_fraud_rate_threshold: float = Field(default=0.05, env="ALERT_FRAUD_RATE_THRESHOLD")  # 5%


class Settings(BaseSettings):
    """Main application settings."""
    
    # App info
    app_name: str = Field(default="Enterprise Fraud Detection API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # Data directories
    data_dir: str = Field(default="data", env="DATA_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
    
    @validator('environment')
    def valid_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings